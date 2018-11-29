import time
import argparse
from datetime import datetime
import logging
import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from models import NavCnnModel, NavCnnRnnModel, NavCnnRnnMultModel, NavPlannerControllerModel
from data import EqaDataLoader
from metrics import NavMetric
from models import MaskedNLLCriterion
from models import get_state, ensure_shared_grads
from data import load_vocab
from torch.autograd import Variable
from tqdm import tqdm
import time

torch.backends.cudnn.enabled = False

################################################################################################
#make models trained in pytorch 4 compatible with earlier pytorch versions
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

################################################################################################

def eval(rank, args, shared_model):

    torch.cuda.set_device(args.gpus.index(args.gpus[rank % len(args.gpus)]))

    if args.model_type == 'cnn':

        model_kwargs = {}
        model = NavCnnModel(**model_kwargs)

    elif args.model_type == 'cnn+q':

        model_kwargs = {
            'question_input': True,
            'question_vocab': load_vocab(args.vocab_json)
        }
        model = NavCnnModel(**model_kwargs)

    elif args.model_type == 'lstm':

        model_kwargs = {}
        model = NavCnnRnnModel(**model_kwargs)

    elif args.model_type == 'lstm+q':

        model_kwargs = {
            'question_input': True,
            'question_vocab': load_vocab(args.vocab_json)
        }
        model = NavCnnRnnModel(**model_kwargs)

    elif args.model_type == 'lstm-mult+q':

        model_kwargs = {
            'question_input': True,
            'question_vocab': load_vocab(args.vocab_json)
        }
        model = NavCnnRnnMultModel(**model_kwargs)

    elif args.model_type == 'pacman':

        model_kwargs = {'question_vocab': load_vocab(args.vocab_json)}
        model = NavPlannerControllerModel(**model_kwargs)

    else:

        exit()

    eval_loader_kwargs = {
        'questions_h5': getattr(args, args.eval_split + '_h5'),
        'data_json': args.data_json,
        'vocab': args.vocab_json,
        'target_obj_conn_map_dir': args.target_obj_conn_map_dir,
        'map_resolution': args.map_resolution,
        'batch_size': 1,
        'input_type': args.model_type,
        'num_frames': 5,
        'split': args.eval_split,
        'max_threads_per_gpu': args.max_threads_per_gpu,
        'gpu_id': args.gpus[rank % len(args.gpus)],
        'to_cache': False,
        'overfit': args.overfit,
        'max_controller_actions': args.max_controller_actions,
    }

    eval_loader = EqaDataLoader(**eval_loader_kwargs)
    print('eval_loader has %d samples' % len(eval_loader.dataset))
    logging.info("EVAL: eval_loader has {} samples".format(len(eval_loader.dataset)))

    args.output_log_path = os.path.join(args.log_dir,
                                        'eval_' + str(rank) + '.json')

    t, epoch, best_eval_acc = 0, 0, 0.0

    max_epochs = args.max_epochs
    if args.mode == 'eval':
        max_epochs = 1
    while epoch < int(max_epochs):

        invalids = []

        model.load_state_dict(shared_model.state_dict())
        model.eval()

        # that's a lot of numbers
        metrics = NavMetric(
            info={'split': args.eval_split,
                  'thread': rank},
            metric_names=[
                'd_0_10', 'd_0_30', 'd_0_50', 'd_T_10', 'd_T_30', 'd_T_50',
                'd_D_10', 'd_D_30', 'd_D_50', 'd_min_10', 'd_min_30',
                'd_min_50', 'r_T_10', 'r_T_30', 'r_T_50', 'r_e_10', 'r_e_30',
                'r_e_50', 'stop_10', 'stop_30', 'stop_50', 'ep_len_10',
                'ep_len_30', 'ep_len_50'
            ],
            log_json=args.output_log_path)

        if 'cnn' in args.model_type:

            done = False

            while done == False:

                for batch in tqdm(eval_loader):

                    model.load_state_dict(shared_model.state_dict())
                    model.cuda()

                    idx, questions, _, img_feats, actions_in, actions_out, action_length = batch
                    metrics_slug = {}

                    # evaluate at multiple initializations
                    for i in [10, 30, 50]:

                        t += 1

                        if action_length[0] + 1 - i - 5 < 0:
                            invalids.append(idx[0])
                            continue

                        ep_inds = [
                            x for x in range(action_length[0] + 1 - i - 5,
                                             action_length[0] + 1 - i)
                        ]

                        sub_img_feats = torch.index_select(
                            img_feats, 1, torch.LongTensor(ep_inds))

                        init_pos = eval_loader.dataset.episode_pos_queue[
                            ep_inds[-1]]

                        h3d = eval_loader.dataset.episode_house

                        h3d.env.reset(
                            x=init_pos[0], y=init_pos[2], yaw=init_pos[3])

                        init_dist_to_target = h3d.get_dist_to_target(
                            h3d.env.cam.pos)
                        if init_dist_to_target < 0:  # unreachable
                            invalids.append(idx[0])
                            continue

                        sub_img_feats_var = Variable(sub_img_feats.cuda())
                        if '+q' in args.model_type:
                            questions_var = Variable(questions.cuda())

                        # sample actions till max steps or <stop>
                        # max no. of actions = 100

                        episode_length = 0
                        episode_done = True

                        dists_to_target, pos_queue, actions = [
                            init_dist_to_target
                        ], [init_pos], []

                        for step in range(args.max_episode_length):

                            episode_length += 1

                            if '+q' in args.model_type:
                                scores = model(sub_img_feats_var,
                                               questions_var)
                            else:
                                scores = model(sub_img_feats_var)

                            prob = F.softmax(scores, dim=1)

                            action = int(prob.max(1)[1].data.cpu().numpy()[0])

                            actions.append(action)

                            img, _, episode_done = h3d.step(action)

                            episode_done = episode_done or episode_length >= args.max_episode_length

                            img = torch.from_numpy(img.transpose(
                                2, 0, 1)).float() / 255.0
                            img_feat_var = eval_loader.dataset.cnn(
                                Variable(img.view(1, 3, 224, 224)
                                         .cuda())).view(1, 1, 3200)
                            sub_img_feats_var = torch.cat(
                                [sub_img_feats_var, img_feat_var], dim=1)
                            sub_img_feats_var = sub_img_feats_var[:, -5:, :]

                            dists_to_target.append(
                                h3d.get_dist_to_target(h3d.env.cam.pos))
                            pos_queue.append([
                                h3d.env.cam.pos.x, h3d.env.cam.pos.y,
                                h3d.env.cam.pos.z, h3d.env.cam.yaw
                            ])

                            if episode_done == True:
                                break

                        # compute stats
                        metrics_slug['d_0_' + str(i)] = dists_to_target[0]
                        metrics_slug['d_T_' + str(i)] = dists_to_target[-1]
                        metrics_slug['d_D_' + str(
                            i)] = dists_to_target[0] - dists_to_target[-1]
                        metrics_slug['d_min_' + str(i)] = np.array(
                            dists_to_target).min()
                        metrics_slug['ep_len_' + str(i)] = episode_length
                        if action == 3:
                            metrics_slug['stop_' + str(i)] = 1
                        else:
                            metrics_slug['stop_' + str(i)] = 0
                        inside_room = []
                        for p in pos_queue:
                            inside_room.append(
                                h3d.is_inside_room(
                                    p, eval_loader.dataset.target_room))
                        if inside_room[-1] == True:
                            metrics_slug['r_T_' + str(i)] = 1
                        else:
                            metrics_slug['r_T_' + str(i)] = 0
                        if any([x == True for x in inside_room]) == True:
                            metrics_slug['r_e_' + str(i)] = 1
                        else:
                            metrics_slug['r_e_' + str(i)] = 0

                    # collate and update metrics
                    metrics_list = []
                    for i in metrics.metric_names:
                        if i not in metrics_slug:
                            metrics_list.append(metrics.metrics[
                                metrics.metric_names.index(i)][0])
                        else:
                            metrics_list.append(metrics_slug[i])

                    # update metrics
                    metrics.update(metrics_list)

                print(metrics.get_stat_string(mode=0))
                print('invalids', len(invalids))
                logging.info("EVAL: metrics: {}".format(metrics.get_stat_string(mode=0)))
                logging.info("EVAL: invalids: {}".format(len(invalids)))

               # del h3d
                eval_loader.dataset._load_envs()
                if len(eval_loader.dataset.pruned_env_set) == 0:
                    done = True

        elif 'lstm' in args.model_type:

            done = False

            while done == False:

                if args.overfit:
                    metrics = NavMetric(
                        info={'split': args.eval_split,
                              'thread': rank},
                        metric_names=[
                            'd_0_10', 'd_0_30', 'd_0_50', 'd_T_10', 'd_T_30', 'd_T_50',
                            'd_D_10', 'd_D_30', 'd_D_50', 'd_min_10', 'd_min_30',
                            'd_min_50', 'r_T_10', 'r_T_30', 'r_T_50', 'r_e_10', 'r_e_30',
                            'r_e_50', 'stop_10', 'stop_30', 'stop_50', 'ep_len_10',
                            'ep_len_30', 'ep_len_50'
                        ],
                        log_json=args.output_log_path)

                for batch in tqdm(eval_loader):

                    model.load_state_dict(shared_model.state_dict())
                    model.cuda()

                    idx, questions, answer, _, actions_in, actions_out, action_lengths, _ = batch
                    question_var = Variable(questions.cuda())
                    metrics_slug = {}

                    # evaluate at multiple initializations
                    for i in [10, 30, 50]:

                        t += 1

                        if action_lengths[0] - 1 - i < 0:
                            invalids.append([idx[0], i])
                            continue

                        h3d = eval_loader.dataset.episode_house

                        # forward through lstm till spawn
                        if len(eval_loader.dataset.episode_pos_queue[:-i]
                               ) > 0:
                            images = eval_loader.dataset.get_frames(
                                h3d,
                                eval_loader.dataset.episode_pos_queue[:-i],
                                preprocess=True)
                            raw_img_feats = eval_loader.dataset.cnn(
                                Variable(torch.FloatTensor(images).cuda()))

                            actions_in_pruned = actions_in[:, :
                                                           action_lengths[0] -
                                                           i]
                            actions_in_var = Variable(actions_in_pruned.cuda())
                            action_lengths_pruned = action_lengths.clone(
                            ).fill_(action_lengths[0] - i)
                            img_feats_var = raw_img_feats.view(1, -1, 3200)

                            if '+q' in args.model_type:
                                scores, hidden = model(
                                    img_feats_var, question_var,
                                    actions_in_var,
                                    action_lengths_pruned.cpu().numpy())
                            else:
                                scores, hidden = model(
                                    img_feats_var, False, actions_in_var,
                                    action_lengths_pruned.cpu().numpy())
                            try:
                                init_pos = eval_loader.dataset.episode_pos_queue[
                                    -i]
                            except:
                                invalids.append([idx[0], i])
                                continue

                            action_in = torch.LongTensor(1, 1).fill_(
                                actions_in[0,
                                           action_lengths[0] - i]).cuda()
                        else:
                            init_pos = eval_loader.dataset.episode_pos_queue[
                                -i]
                            hidden = model.nav_rnn.init_hidden(1)
                            action_in = torch.LongTensor(1, 1).fill_(0).cuda()

                        h3d.env.reset(
                            x=init_pos[0], y=init_pos[2], yaw=init_pos[3])

                        init_dist_to_target = h3d.get_dist_to_target(
                            h3d.env.cam.pos)
                        if init_dist_to_target < 0:  # unreachable
                            invalids.append([idx[0], i])
                            continue

                        img = h3d.env.render()
                        img = torch.from_numpy(img.transpose(
                            2, 0, 1)).float() / 255.0
                        img_feat_var = eval_loader.dataset.cnn(
                            Variable(img.view(1, 3, 224, 224).cuda())).view(
                                1, 1, 3200)

                        episode_length = 0
                        episode_done = True

                        dists_to_target, pos_queue, actions = [
                            init_dist_to_target
                        ], [init_pos], []
                        actual_pos_queue = [(h3d.env.cam.pos.x, h3d.env.cam.pos.z, h3d.env.cam.yaw)]

                        for step in range(args.max_episode_length):

                            episode_length += 1

                            if '+q' in args.model_type:
                                scores, hidden = model(
                                    img_feat_var,
                                    question_var,
                                    Variable(action_in),
                                    False,
                                    hidden=hidden,
                                    step=True)
                            else:
                                scores, hidden = model(
                                    img_feat_var,
                                    False,
                                    Variable(action_in),
                                    False,
                                    hidden=hidden,
                                    step=True)

                            prob = F.softmax(scores, dim=1)

                            action = int(prob.max(1)[1].data.cpu().numpy()[0])

                            actions.append(action)

                            img, _, episode_done = h3d.step(action)

                            episode_done = episode_done or episode_length >= args.max_episode_length

                            img = torch.from_numpy(img.transpose(
                                2, 0, 1)).float() / 255.0
                            img_feat_var = eval_loader.dataset.cnn(
                                Variable(img.view(1, 3, 224, 224)
                                         .cuda())).view(1, 1, 3200)

                            action_in = torch.LongTensor(
                                1, 1).fill_(action + 1).cuda()

                            dists_to_target.append(
                                h3d.get_dist_to_target(h3d.env.cam.pos))
                            pos_queue.append([
                                h3d.env.cam.pos.x, h3d.env.cam.pos.y,
                                h3d.env.cam.pos.z, h3d.env.cam.yaw
                            ])

                            if episode_done == True:
                                break

                            actual_pos_queue.append([
                                h3d.env.cam.pos.x, h3d.env.cam.pos.z, h3d.env.cam.yaw])

                        # compute stats
                        metrics_slug['d_0_' + str(i)] = dists_to_target[0]
                        metrics_slug['d_T_' + str(i)] = dists_to_target[-1]
                        metrics_slug['d_D_' + str(
                            i)] = dists_to_target[0] - dists_to_target[-1]
                        metrics_slug['d_min_' + str(i)] = np.array(
                            dists_to_target).min()
                        metrics_slug['ep_len_' + str(i)] = episode_length
                        if action == 3:
                            metrics_slug['stop_' + str(i)] = 1
                        else:
                            metrics_slug['stop_' + str(i)] = 0
                        inside_room = []
                        for p in pos_queue:
                            inside_room.append(
                                h3d.is_inside_room(
                                    p, eval_loader.dataset.target_room))
                        if inside_room[-1] == True:
                            metrics_slug['r_T_' + str(i)] = 1
                        else:
                            metrics_slug['r_T_' + str(i)] = 0
                        if any([x == True for x in inside_room]) == True:
                            metrics_slug['r_e_' + str(i)] = 1
                        else:
                            metrics_slug['r_e_' + str(i)] = 0

                    # collate and update metrics
                    metrics_list = []
                    for i in metrics.metric_names:
                        if i not in metrics_slug:
                            metrics_list.append(metrics.metrics[
                                metrics.metric_names.index(i)][0])
                        else:
                            metrics_list.append(metrics_slug[i])

                    # update metrics
                    metrics.update(metrics_list)

                print(metrics.get_stat_string(mode=0))
                print('invalids', len(invalids))
                logging.info("EVAL: init_steps: {} metrics: {}".format(i, metrics.get_stat_string(mode=0)))
                logging.info("EVAL: init_steps: {} invalids: {}".format(i, len(invalids)))

                # del h3d
                eval_loader.dataset._load_envs()
                print("eval_loader pruned_env_set len: {}".format(len(eval_loader.dataset.pruned_env_set)))
                logging.info("eval_loader pruned_env_set len: {}".format(len(eval_loader.dataset.pruned_env_set)))
                assert len(eval_loader.dataset.pruned_env_set) > 0
                if len(eval_loader.dataset.pruned_env_set) == 0:
                    done = True

        elif 'pacman' in args.model_type:

            done = False

            while done == False:
                if args.overfit:
                    metrics = NavMetric(
                        info={'split': args.eval_split,
                              'thread': rank},
                        metric_names=[
                            'd_0_10', 'd_0_30', 'd_0_50', 'd_T_10', 'd_T_30', 'd_T_50',
                            'd_D_10', 'd_D_30', 'd_D_50', 'd_min_10', 'd_min_30',
                            'd_min_50', 'r_T_10', 'r_T_30', 'r_T_50', 'r_e_10', 'r_e_30',
                            'r_e_50', 'stop_10', 'stop_30', 'stop_50', 'ep_len_10',
                            'ep_len_30', 'ep_len_50'
                        ],
                        log_json=args.output_log_path)

                for batch in tqdm(eval_loader):

                    model.load_state_dict(shared_model.state_dict())
                    model.cuda()

                    idx, question, answer, actions, action_length = batch
                    metrics_slug = {}

                    h3d = eval_loader.dataset.episode_house

                    # evaluate at multiple initializations
                    for i in [10, 30, 50]:

                        t += 1

                        if i > action_length[0]:
                            invalids.append([idx[0], i])
                            continue

                        question_var = Variable(question.cuda())

                        controller_step = False
                        planner_hidden = model.planner_nav_rnn.init_hidden(1)

                        # get hierarchical action history
                        (
                            planner_actions_in, planner_img_feats,
                            controller_step, controller_action_in,
                            controller_img_feats, init_pos,
                            controller_action_counter
                        ) = eval_loader.dataset.get_hierarchical_features_till_spawn(
                            actions[0, :action_length[0] + 1].numpy(), i, args.max_controller_actions
                        )

                        planner_actions_in_var = Variable(
                            planner_actions_in.cuda())
                        planner_img_feats_var = Variable(
                            planner_img_feats.cuda())

                        # forward planner till spawn to update hidden state
                        for step in range(planner_actions_in.size(0)):

                            planner_scores, planner_hidden = model.planner_step(
                                question_var, planner_img_feats_var[step]
                                .unsqueeze(0).unsqueeze(0),
                                planner_actions_in_var[step].view(1, 1),
                                planner_hidden
                            )

                        h3d.env.reset(
                            x=init_pos[0], y=init_pos[2], yaw=init_pos[3])

                        init_dist_to_target = h3d.get_dist_to_target(
                            h3d.env.cam.pos)
                        if init_dist_to_target < 0:  # unreachable
                            invalids.append([idx[0], i])
                            continue

                        dists_to_target, pos_queue, pred_actions = [
                            init_dist_to_target
                        ], [init_pos], []
                        planner_actions, controller_actions = [], []

                        episode_length = 0
                        if args.max_controller_actions > 1:
                            controller_action_counter = controller_action_counter % args.max_controller_actions
                            controller_action_counter = max(controller_action_counter - 1, 0)
                        else:
                            controller_action_counter = 0

                        first_step = True
                        first_step_is_controller = controller_step
                        planner_step = True
                        action = int(controller_action_in)

                        for step in range(args.max_episode_length):
                            if not first_step:
                                img = torch.from_numpy(img.transpose(
                                    2, 0, 1)).float() / 255.0
                                img_feat_var = eval_loader.dataset.cnn(
                                    Variable(img.view(1, 3, 224,
                                                      224).cuda())).view(
                                                          1, 1, 3200)
                            else:
                                img_feat_var = Variable(controller_img_feats.cuda()).view(1, 1, 3200)

                            if not first_step or first_step_is_controller:
                                # query controller to continue or not
                                controller_action_in = Variable(
                                    torch.LongTensor(1, 1).fill_(action).cuda())
                                controller_scores = model.controller_step(
                                    img_feat_var, controller_action_in,
                                    planner_hidden[0])

                                prob = F.softmax(controller_scores, dim=1)
                                controller_action = int(
                                    prob.max(1)[1].data.cpu().numpy()[0])

                                if controller_action == 1 and controller_action_counter < args.max_controller_actions - 1:
                                    controller_action_counter += 1
                                    planner_step = False
                                else:
                                    controller_action_counter = 0
                                    planner_step = True
                                    controller_action = 0

                                controller_actions.append(controller_action)
                                first_step = False

                            if planner_step:
                                if not first_step:
                                    action_in = torch.LongTensor(
                                        1, 1).fill_(action + 1).cuda()
                                    planner_scores, planner_hidden = model.planner_step(
                                        question_var, img_feat_var,
                                        Variable(action_in), planner_hidden)

                                prob = F.softmax(planner_scores, dim=1)
                                action = int(
                                    prob.max(1)[1].data.cpu().numpy()[0])
                                planner_actions.append(action)

                            episode_done = action == 3 or episode_length >= args.max_episode_length

                            episode_length += 1
                            dists_to_target.append(
                                h3d.get_dist_to_target(h3d.env.cam.pos))
                            pos_queue.append([
                                h3d.env.cam.pos.x, h3d.env.cam.pos.y,
                                h3d.env.cam.pos.z, h3d.env.cam.yaw
                            ])

                            if episode_done:
                                break

                            img, _, _ = h3d.step(action)
                            first_step = False

                        # compute stats
                        metrics_slug['d_0_' + str(i)] = dists_to_target[0]
                        metrics_slug['d_T_' + str(i)] = dists_to_target[-1]
                        metrics_slug['d_D_' + str(
                            i)] = dists_to_target[0] - dists_to_target[-1]
                        metrics_slug['d_min_' + str(i)] = np.array(
                            dists_to_target).min()
                        metrics_slug['ep_len_' + str(i)] = episode_length
                        if action == 3:
                            metrics_slug['stop_' + str(i)] = 1
                        else:
                            metrics_slug['stop_' + str(i)] = 0
                        inside_room = []
                        for p in pos_queue:
                            inside_room.append(
                                h3d.is_inside_room(
                                    p, eval_loader.dataset.target_room))
                        if inside_room[-1] == True:
                            metrics_slug['r_T_' + str(i)] = 1
                        else:
                            metrics_slug['r_T_' + str(i)] = 0
                        if any([x == True for x in inside_room]) == True:
                            metrics_slug['r_e_' + str(i)] = 1
                        else:
                            metrics_slug['r_e_' + str(i)] = 0

                    # collate and update metrics
                    metrics_list = []
                    for i in metrics.metric_names:
                        if i not in metrics_slug:
                            metrics_list.append(metrics.metrics[
                                metrics.metric_names.index(i)][0])
                        else:
                            metrics_list.append(metrics_slug[i])

                    # update metrics
                    metrics.update(metrics_list)

                try:
                    print(metrics.get_stat_string(mode=0))
                    logging.info("EVAL: metrics: {}".format(metrics.get_stat_string(mode=0)))
                except:
                    pass

                print('epoch', epoch)
                print('invalids', len(invalids))
                logging.info("EVAL: epoch {}".format(epoch))
                logging.info("EVAL: invalids {}".format(invalids))

                # del h3d
                eval_loader.dataset._load_envs()
                if len(eval_loader.dataset.pruned_env_set) == 0:
                    done = True

        epoch += 1

        # checkpoint if best val loss
        if metrics.metrics[8][0] > best_eval_acc:  # d_D_50
            best_eval_acc = metrics.metrics[8][0]
            if epoch % args.eval_every == 0 and args.log == True:
                metrics.dump_log()

                model_state = get_state(model)

                aad = dict(args.__dict__)
                ad = {}
                for i in aad:
                    if i[0] != '_':
                        ad[i] = aad[i]

                checkpoint = {'args': ad, 'state': model_state, 'epoch': epoch}

                checkpoint_path = '%s/epoch_%d_d_D_50_%.04f.pt' % (
                    args.checkpoint_dir, epoch, best_eval_acc)
                print('Saving checkpoint to %s' % checkpoint_path)
                logging.info("EVAL: Saving checkpoint to {}".format(checkpoint_path))
                torch.save(checkpoint, checkpoint_path)

        print('[best_eval_d_D_50:%.04f]' % best_eval_acc)
        logging.info("EVAL: [best_eval_d_D_50:{:.04f}]".format(best_eval_acc))

        eval_loader.dataset._load_envs(start_idx=0, in_order=True)


def train(rank, args, shared_model):
    torch.cuda.set_device(args.gpus.index(args.gpus[rank % len(args.gpus)]))

    if args.model_type == 'cnn':

        model_kwargs = {}
        model = NavCnnModel(**model_kwargs)

    elif args.model_type == 'cnn+q':

        model_kwargs = {
            'question_input': True,
            'question_vocab': load_vocab(args.vocab_json)
        }
        model = NavCnnModel(**model_kwargs)

    elif args.model_type == 'lstm':

        model_kwargs = {}
        model = NavCnnRnnModel(**model_kwargs)

    elif args.model_type == 'lstm-mult+q':

        model_kwargs = {
            'question_input': True,
            'question_vocab': load_vocab(args.vocab_json)
        }
        model = NavCnnRnnMultModel(**model_kwargs)

    elif args.model_type == 'lstm+q':

        model_kwargs = {
            'question_input': True,
            'question_vocab': load_vocab(args.vocab_json)
        }
        model = NavCnnRnnModel(**model_kwargs)

    elif args.model_type == 'pacman':

        model_kwargs = {'question_vocab': load_vocab(args.vocab_json)}
        model = NavPlannerControllerModel(**model_kwargs)

    else:

        exit()

    lossFn = torch.nn.CrossEntropyLoss().cuda()

    optim = torch.optim.Adamax(
        filter(lambda p: p.requires_grad, shared_model.parameters()),
        lr=args.learning_rate)

    train_loader_kwargs = {
        'questions_h5': args.train_h5,
        'data_json': args.data_json,
        'vocab': args.vocab_json,
        'batch_size': args.batch_size,
        'input_type': args.model_type,
        'num_frames': 5,
        'map_resolution': args.map_resolution,
        'split': 'train',
        'max_threads_per_gpu': args.max_threads_per_gpu,
        'gpu_id': args.gpus[rank % len(args.gpus)],
        'to_cache': args.cache,
        'overfit': args.overfit,
        'max_controller_actions': args.max_controller_actions,
        'max_actions': args.max_actions
    }

    args.output_log_path = os.path.join(args.log_dir,
                                        'train_' + str(rank) + '.json')

    if 'pacman' in args.model_type:

        metrics = NavMetric(
            info={'split': 'train',
                  'thread': rank},
            metric_names=['planner_loss', 'controller_loss'],
            log_json=args.output_log_path)

    else:

        metrics = NavMetric(
            info={'split': 'train',
                  'thread': rank},
            metric_names=['loss'],
            log_json=args.output_log_path)

    train_loader = EqaDataLoader(**train_loader_kwargs)

    print('train_loader has %d samples' % len(train_loader.dataset))
    logging.info('TRAIN: train loader has {} samples'.format(len(train_loader.dataset)))

    t, epoch = 0, 0

    while epoch < int(args.max_epochs):

        if 'cnn' in args.model_type:

            done = False
            all_envs_loaded = train_loader.dataset._check_if_all_envs_loaded()

            while done == False:

                for batch in train_loader:

                    t += 1

                    model.load_state_dict(shared_model.state_dict())
                    model.train()
                    model.cuda()

                    idx, questions, _, img_feats, _, actions_out, _ = batch

                    img_feats_var = Variable(img_feats.cuda())
                    if '+q' in args.model_type:
                        questions_var = Variable(questions.cuda())
                    actions_out_var = Variable(actions_out.cuda())

                    if '+q' in args.model_type:
                        scores = model(img_feats_var, questions_var)
                    else:
                        scores = model(img_feats_var)

                    loss = lossFn(scores, actions_out_var)

                    # zero grad
                    optim.zero_grad()

                    # update metrics
                    metrics.update([loss.data[0]])

                    # backprop and update
                    loss.backward()

                    ensure_shared_grads(model.cpu(), shared_model)
                    optim.step()

                    if t % args.print_every == 0:
                        print(metrics.get_stat_string())
                        logging.info("TRAIN: metrics: {}".format(metrics.get_stat_string()))
                        if args.log == True:
                            metrics.dump_log()

                    print('[CHECK][Cache:%d][Total:%d]' %
                          (len(train_loader.dataset.img_data_cache),
                           len(train_loader.dataset.env_list)))
                    logging.info('TRAIN: [CHECK][Cache:{}][Total:{}]'.format(
                        len(train_loader.dataset.img_data_cache), len(train_loader.dataset.env_list)))

                if all_envs_loaded == False:
                    train_loader.dataset._load_envs(in_order=True)
                    if len(train_loader.dataset.pruned_env_set) == 0:
                        done = True
                        if args.cache == False:
                            train_loader.dataset._load_envs(
                                start_idx=0, in_order=True)
                else:
                    done = True

        elif 'lstm' in args.model_type:

            lossFn = MaskedNLLCriterion().cuda()

            done = False
            all_envs_loaded = train_loader.dataset._check_if_all_envs_loaded()
            total_times = []
            while done == False:

                start_time = time.time()
                for batch in train_loader:

                    t += 1


                    model.load_state_dict(shared_model.state_dict())
                    model.train()
                    model.cuda()

                    idx, questions, _, img_feats, actions_in, actions_out, action_lengths, masks = batch

                    img_feats_var = Variable(img_feats.cuda())
                    if '+q' in args.model_type:
                        questions_var = Variable(questions.cuda())
                    actions_in_var = Variable(actions_in.cuda())
                    actions_out_var = Variable(actions_out.cuda())
                    action_lengths = action_lengths.cuda()
                    masks_var = Variable(masks.cuda())

                    action_lengths, perm_idx = action_lengths.sort(
                        0, descending=True)

                    img_feats_var = img_feats_var[perm_idx]
                    if '+q' in args.model_type:
                        questions_var = questions_var[perm_idx]
                    actions_in_var = actions_in_var[perm_idx]
                    actions_out_var = actions_out_var[perm_idx]
                    masks_var = masks_var[perm_idx]

                    if '+q' in args.model_type:
                        scores, hidden = model(img_feats_var, questions_var,
                                               actions_in_var,
                                               action_lengths.cpu().numpy())
                    else:
                        scores, hidden = model(img_feats_var, False,
                                               actions_in_var,
                                               action_lengths.cpu().numpy())

                    #block out masks
                    if args.curriculum:
                        curriculum_length = (epoch+1)*5
                        for i, action_length in enumerate(action_lengths):
                            if action_length - curriculum_length > 0:
                                masks_var[i, :action_length-curriculum_length] = 0

                    logprob = F.log_softmax(scores, dim=1)
                    loss = lossFn(
                        logprob, actions_out_var[:, :action_lengths.max()]
                        .contiguous().view(-1, 1),
                        masks_var[:, :action_lengths.max()].contiguous().view(
                            -1, 1))

                    # zero grad
                    optim.zero_grad()

                    # update metrics
                    metrics.update([loss.data[0]])
                    logging.info("TRAIN LSTM loss: {:.6f}".format(loss.data[0]))

                    # backprop and update
                    loss.backward()

                    ensure_shared_grads(model.cpu(), shared_model)
                    optim.step()

                    if t % args.print_every == 0:
                        print(metrics.get_stat_string())
                        logging.info("TRAIN: metrics: {}".format(metrics.get_stat_string()))
                        if args.log == True:
                            metrics.dump_log()

                    print('[CHECK][Cache:%d][Total:%d]' %
                          (len(train_loader.dataset.img_data_cache),
                           len(train_loader.dataset.env_list)))
                    logging.info('TRAIN: [CHECK][Cache:{}][Total:{}]'.format(
                        len(train_loader.dataset.img_data_cache), len(train_loader.dataset.env_list)))


                if all_envs_loaded == False:
                    train_loader.dataset._load_envs(in_order=True)
                    if len(train_loader.dataset.pruned_env_set) == 0:
                        done = True
                        if args.cache == False:
                            train_loader.dataset._load_envs(
                                start_idx=0, in_order=True)
                else:
                    done = True

        elif 'pacman' in args.model_type:

            planner_lossFn = MaskedNLLCriterion().cuda()
            controller_lossFn = MaskedNLLCriterion().cuda()

            done = False
            all_envs_loaded = train_loader.dataset._check_if_all_envs_loaded()

            while done == False:

                for batch in train_loader:

                    t += 1

                    model.load_state_dict(shared_model.state_dict())
                    model.train()
                    model.cuda()

                    idx, questions, _, planner_img_feats, planner_actions_in, \
                        planner_actions_out, planner_action_lengths, planner_masks, \
                        controller_img_feats, controller_actions_in, planner_hidden_idx, \
                        controller_outs, controller_action_lengths, controller_masks = batch

                    questions_var = Variable(questions.cuda())

                    planner_img_feats_var = Variable(planner_img_feats.cuda())
                    planner_actions_in_var = Variable(
                        planner_actions_in.cuda())
                    planner_actions_out_var = Variable(
                        planner_actions_out.cuda())
                    planner_action_lengths = planner_action_lengths.cuda()
                    planner_masks_var = Variable(planner_masks.cuda())

                    controller_img_feats_var = Variable(
                        controller_img_feats.cuda())
                    controller_actions_in_var = Variable(
                        controller_actions_in.cuda())
                    planner_hidden_idx_var = Variable(
                        planner_hidden_idx.cuda())
                    controller_outs_var = Variable(controller_outs.cuda())
                    controller_action_lengths = controller_action_lengths.cuda(
                    )
                    controller_masks_var = Variable(controller_masks.cuda())

                    planner_action_lengths, perm_idx = planner_action_lengths.sort(
                        0, descending=True)

                    questions_var = questions_var[perm_idx]

                    planner_img_feats_var = planner_img_feats_var[perm_idx]
                    planner_actions_in_var = planner_actions_in_var[perm_idx]
                    planner_actions_out_var = planner_actions_out_var[perm_idx]
                    planner_masks_var = planner_masks_var[perm_idx]

                    controller_img_feats_var = controller_img_feats_var[
                        perm_idx]
                    controller_actions_in_var = controller_actions_in_var[
                        perm_idx]
                    controller_outs_var = controller_outs_var[perm_idx]
                    planner_hidden_idx_var = planner_hidden_idx_var[perm_idx]
                    controller_action_lengths = controller_action_lengths[
                        perm_idx]
                    controller_masks_var = controller_masks_var[perm_idx]

                    planner_scores, controller_scores, planner_hidden = model(
                        questions_var, planner_img_feats_var,
                        planner_actions_in_var,
                        planner_action_lengths.cpu().numpy(),
                        planner_hidden_idx_var, controller_img_feats_var,
                        controller_actions_in_var, controller_action_lengths)

                    planner_logprob = F.log_softmax(planner_scores, dim=1)
                    controller_logprob = F.log_softmax(
                        controller_scores, dim=1)

                    planner_loss = planner_lossFn(
                        planner_logprob,
                        planner_actions_out_var[:, :planner_action_lengths.max(
                        )].contiguous().view(-1, 1),
                        planner_masks_var[:, :planner_action_lengths.max()]
                        .contiguous().view(-1, 1))

                    controller_loss = controller_lossFn(
                        controller_logprob,
                        controller_outs_var[:, :controller_action_lengths.max(
                        )].contiguous().view(-1, 1),
                        controller_masks_var[:, :controller_action_lengths.max(
                        )].contiguous().view(-1, 1))

                    # zero grad
                    optim.zero_grad()

                    # update metrics
                    metrics.update(
                        [planner_loss.data[0], controller_loss.data[0]])
                    logging.info("TRAINING PACMAN planner-loss: {:.6f} controller-loss: {:.6f}".format(
                        planner_loss.data[0], controller_loss.data[0]))

                    # backprop and update
                    if args.max_controller_actions == 1:
                        (planner_loss).backward()
                    else:
                        (planner_loss + controller_loss).backward()

                    ensure_shared_grads(model.cpu(), shared_model)
                    optim.step()

                    if t % args.print_every == 0:
                        print(metrics.get_stat_string())
                        logging.info("TRAIN: metrics: {}".format(metrics.get_stat_string()))
                        if args.log == True:
                            metrics.dump_log()

                    print('[CHECK][Cache:%d][Total:%d]' %
                          (len(train_loader.dataset.img_data_cache),
                           len(train_loader.dataset.env_list)))
                    logging.info('TRAIN: [CHECK][Cache:{}][Total:{}]'.format(
                        len(train_loader.dataset.img_data_cache), len(train_loader.dataset.env_list)))

                if all_envs_loaded == False:
                    train_loader.dataset._load_envs(in_order=True)
                    if len(train_loader.dataset.pruned_env_set) == 0:
                        done = True
                        if args.cache == False:
                            train_loader.dataset._load_envs(
                                start_idx=0, in_order=True)
                else:
                    done = True

        epoch += 1

        if epoch % args.save_every == 0:

            model_state = get_state(model)
            optimizer_state = optim.state_dict()

            aad = dict(args.__dict__)
            ad = {}
            for i in aad:
                if i[0] != '_':
                    ad[i] = aad[i]

            checkpoint = {'args': ad,
                          'state': model_state,
                          'epoch': epoch,
                          'optimizer': optimizer_state}

            checkpoint_path = '%s/epoch_%d_thread_%d.pt' % (
                args.checkpoint_dir, epoch, rank)
            print('Saving checkpoint to %s' % checkpoint_path)
            logging.info("TRAIN: Saving checkpoint to {}".format(checkpoint_path))
            torch.save(checkpoint, checkpoint_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data params
    parser.add_argument('-train_h5', default='data/train.h5')
    parser.add_argument('-val_h5', default='data/val.h5')
    parser.add_argument('-test_h5', default='data/test.h5')
    parser.add_argument('-data_json', default='data/data.json')
    parser.add_argument('-vocab_json', default='data/vocab.json')

    parser.add_argument(
        '-target_obj_conn_map_dir',
        default='data/target-obj-conn-maps/500')
    parser.add_argument('-map_resolution', default=500, type=int)

    parser.add_argument(
        '-mode',
        default='train+eval',
        type=str,
        choices=['train', 'eval', 'train+eval'])
    parser.add_argument('-eval_split', default='val', type=str)

    # model details
    parser.add_argument(
        '-model_type',
        default='cnn',
        choices=['cnn', 'cnn+q', 'lstm', 'lstm+q', 'lstm-mult+q', 'pacman'])
    parser.add_argument('-max_episode_length', default=100, type=int)
    parser.add_argument('-curriculum', default=0, type=int)

    # optim params
    parser.add_argument('-batch_size', default=20, type=int)
    parser.add_argument('-learning_rate', default=1e-3, type=float)
    parser.add_argument('-max_epochs', default=1000, type=int)
    parser.add_argument('-overfit', default=False, action='store_true')

    # bookkeeping
    parser.add_argument('-print_every', default=5, type=int)
    parser.add_argument('-eval_every', default=1, type=int)
    parser.add_argument('-save_every', default=1000, type=int) #optional if you would like to save specific epochs as opposed to relying on the eval thread
    parser.add_argument('-identifier', default='cnn')
    parser.add_argument('-num_processes', default=1, type=int)
    parser.add_argument('-max_threads_per_gpu', default=10, type=int)

    # checkpointing
    parser.add_argument('-checkpoint_path', default=False)
    parser.add_argument('-checkpoint_dir', default='checkpoints/nav/')
    parser.add_argument('-log_dir', default='logs/nav/')
    parser.add_argument('-log', default=False, action='store_true')
    parser.add_argument('-cache', default=False, action='store_true')
    parser.add_argument('-max_controller_actions', type=int, default=5)
    parser.add_argument('-max_actions', type=int)
    args = parser.parse_args()

    args.time_id = time.strftime("%m_%d_%H:%M")

    #MAX_CONTROLLER_ACTIONS = args.max_controller_actions

    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    if args.curriculum:
        assert 'lstm' in args.model_type #TODO: Finish implementing curriculum for other model types

    logging.basicConfig(filename=os.path.join(args.log_dir, "run_{}.log".format(
                                                str(datetime.now()).replace(' ', '_'))),
                        level=logging.INFO,
                        format='%(asctime)-15s %(message)s')

    try:
        args.gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        args.gpus = [int(x) for x in args.gpus]
    except KeyError:
        print("CPU not supported")
        logging.info("CPU not supported")
        exit()

    if args.checkpoint_path != False:

        print('Loading checkpoint from %s' % args.checkpoint_path)
        logging.info("Loading checkpoint from {}".format(args.checkpoint_path))

        args_to_keep = ['model_type']

        checkpoint = torch.load(args.checkpoint_path, map_location={
            'cuda:0': 'cpu'
        })

        for i in args.__dict__:
            if i not in args_to_keep:
                checkpoint['args'][i] = args.__dict__[i]

        args = type('new_dict', (object, ), checkpoint['args'])

    args.checkpoint_dir = os.path.join(args.checkpoint_dir,
                                       args.time_id + '_' + args.identifier)
    args.log_dir = os.path.join(args.log_dir,
                                args.time_id + '_' + args.identifier)


    # if set to overfit; set eval_split to train
    if args.overfit == True:
        args.eval_split = 'train'

    print(args.__dict__)
    logging.info(args.__dict__)

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
        os.makedirs(args.log_dir)

    if args.model_type == 'cnn':

        model_kwargs = {}
        shared_model = NavCnnModel(**model_kwargs)

    elif args.model_type == 'cnn+q':

        model_kwargs = {
            'question_input': True,
            'question_vocab': load_vocab(args.vocab_json)
        }
        shared_model = NavCnnModel(**model_kwargs)

    elif args.model_type == 'lstm':

        model_kwargs = {}
        shared_model = NavCnnRnnModel(**model_kwargs)

    elif args.model_type == 'lstm+q':

        model_kwargs = {
            'question_input': True,
            'question_vocab': load_vocab(args.vocab_json)
        }
        shared_model = NavCnnRnnModel(**model_kwargs)

    elif args.model_type == 'pacman':

        model_kwargs = {'question_vocab': load_vocab(args.vocab_json)}
        shared_model = NavPlannerControllerModel(**model_kwargs)

    else:

        exit()

    shared_model.share_memory()

    if args.checkpoint_path != False:
        print('Loading params from checkpoint: %s' % args.checkpoint_path)
        logging.info("Loading params from checkpoint: {}".format(args.checkpoint_path))
        shared_model.load_state_dict(checkpoint['state'])

    if args.mode == 'eval':

        eval(0, args, shared_model)

    elif args.mode == 'train':

        if args.num_processes > 1:
            processes = []
            for rank in range(0, args.num_processes):
                # for rank in range(0, args.num_processes):
                p = mp.Process(target=train, args=(rank, args, shared_model))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

        else:
            train(0, args, shared_model)

    else:
        processes = []

        # Start the eval thread
        p = mp.Process(target=eval, args=(0, args, shared_model))
        p.start()
        processes.append(p)

        # Start the training thread(s)
        for rank in range(1, args.num_processes + 1):
            # for rank in range(0, args.num_processes):
            p = mp.Process(target=train, args=(rank, args, shared_model))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
