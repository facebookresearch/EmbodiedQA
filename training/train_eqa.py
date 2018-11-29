# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import h5py
import time
import argparse
import numpy as np
import os, sys, json
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.autograd import Variable
torch.backends.cudnn.enabled = False
import torch.multiprocessing as mp

from models import NavCnnModel, NavCnnRnnModel, NavPlannerControllerModel, VqaLstmCnnAttentionModel
from data import EqaDataset, EqaDataLoader
from metrics import NavMetric, VqaMetric

from models import MaskedNLLCriterion

from models import get_state, repackage_hidden, ensure_shared_grads
from data import load_vocab, flat_to_hierarchical_actions

def eval(rank, args, shared_nav_model, shared_ans_model):

    torch.cuda.set_device(args.gpus.index(args.gpus[rank % len(args.gpus)]))

    if args.model_type == 'pacman':

        model_kwargs = {'question_vocab': load_vocab(args.vocab_json)}
        nav_model = NavPlannerControllerModel(**model_kwargs)

    else:

        exit()

    model_kwargs = {'vocab': load_vocab(args.vocab_json)}
    ans_model = VqaLstmCnnAttentionModel(**model_kwargs)

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
        'max_controller_actions': args.max_controller_actions,
        'max_actions': args.max_actions
    }

    eval_loader = EqaDataLoader(**eval_loader_kwargs)
    print('eval_loader has %d samples' % len(eval_loader.dataset))

    args.output_nav_log_path = os.path.join(args.log_dir,
                                            'nav_eval_' + str(rank) + '.json')
    args.output_ans_log_path = os.path.join(args.log_dir,
                                            'ans_eval_' + str(rank) + '.json')

    t, epoch, best_eval_acc = 0, 0, 0.0

    while epoch < int(args.max_epochs):

        start_time = time.time()
        invalids = []

        nav_model.load_state_dict(shared_nav_model.state_dict())
        nav_model.eval()

        ans_model.load_state_dict(shared_ans_model.state_dict())
        ans_model.eval()
        ans_model.cuda()

        # that's a lot of numbers
        nav_metrics = NavMetric(
            info={'split': args.eval_split,
                  'thread': rank},
            metric_names=[
                'd_0_10', 'd_0_30', 'd_0_50', 'd_T_10', 'd_T_30', 'd_T_50',
                'd_D_10', 'd_D_30', 'd_D_50', 'd_min_10', 'd_min_30',
                'd_min_50', 'r_T_10', 'r_T_30', 'r_T_50', 'r_e_10', 'r_e_30',
                'r_e_50', 'stop_10', 'stop_30', 'stop_50', 'ep_len_10',
                'ep_len_30', 'ep_len_50'
            ],
            log_json=args.output_nav_log_path)

        vqa_metrics = VqaMetric(
            info={'split': args.eval_split,
                  'thread': rank},
            metric_names=[
                'accuracy_10', 'accuracy_30', 'accuracy_50', 'mean_rank_10',
                'mean_rank_30', 'mean_rank_50', 'mean_reciprocal_rank_10',
                'mean_reciprocal_rank_30', 'mean_reciprocal_rank_50'
            ],
            log_json=args.output_ans_log_path)

        if 'pacman' in args.model_type:

            done = False

            while done == False:

                for batch in tqdm(eval_loader):

                    nav_model.load_state_dict(shared_nav_model.state_dict())
                    nav_model.eval()
                    nav_model.cuda()

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
                        planner_hidden = nav_model.planner_nav_rnn.init_hidden(
                            1)

                        # forward through planner till spawn
                        (
                            planner_actions_in, planner_img_feats,
                            controller_step, controller_action_in,
                            controller_img_feat, init_pos,
                            controller_action_counter
                        ) = eval_loader.dataset.get_hierarchical_features_till_spawn(
                            actions[0, :action_length[0] + 1].numpy(), i, args.max_controller_actions
                        )

                        planner_actions_in_var = Variable(
                            planner_actions_in.cuda())
                        planner_img_feats_var = Variable(
                            planner_img_feats.cuda())

                        for step in range(planner_actions_in.size(0)):

                            planner_scores, planner_hidden = nav_model.planner_step(
                                question_var, planner_img_feats_var[step].view(
                                    1, 1,
                                    3200), planner_actions_in_var[step].view(
                                        1, 1), planner_hidden)

                        if controller_step == True:

                            controller_img_feat_var = Variable(
                                controller_img_feat.cuda())
                            controller_action_in_var = Variable(
                                torch.LongTensor(1, 1).fill_(
                                    int(controller_action_in)).cuda())

                            controller_scores = nav_model.controller_step(
                                controller_img_feat_var.view(1, 1, 3200),
                                controller_action_in_var.view(1, 1),
                                planner_hidden[0])

                            prob = F.softmax(controller_scores, dim=1)
                            controller_action = int(
                                prob.max(1)[1].data.cpu().numpy()[0])

                            if controller_action == 1:
                                controller_step = True
                            else:
                                controller_step = False

                            action = int(controller_action_in)
                            action_in = torch.LongTensor(
                                1, 1).fill_(action + 1).cuda()

                        else:

                            prob = F.softmax(planner_scores, dim=1)
                            action = int(prob.max(1)[1].data.cpu().numpy()[0])

                            action_in = torch.LongTensor(
                                1, 1).fill_(action + 1).cuda()

                        h3d.env.reset(
                            x=init_pos[0], y=init_pos[2], yaw=init_pos[3])

                        init_dist_to_target = h3d.get_dist_to_target(
                            h3d.env.cam.pos)
                        if init_dist_to_target < 0:  # unreachable
                            invalids.append([idx[0], i])
                            continue

                        episode_length = 0
                        episode_done = True
                        controller_action_counter = 0

                        dists_to_target, pos_queue, pred_actions = [
                            init_dist_to_target
                        ], [init_pos], []
                        planner_actions, controller_actions = [], []

                        if action != 3:

                            # take the first step
                            img, _, _ = h3d.step(action)
                            img = torch.from_numpy(img.transpose(
                                2, 0, 1)).float() / 255.0
                            img_feat_var = eval_loader.dataset.cnn(
                                Variable(img.view(1, 3, 224,
                                                  224).cuda())).view(
                                                      1, 1, 3200)

                            for step in range(args.max_episode_length):

                                episode_length += 1

                                if controller_step == False:
                                    planner_scores, planner_hidden = nav_model.planner_step(
                                        question_var, img_feat_var,
                                        Variable(action_in), planner_hidden)

                                    prob = F.softmax(planner_scores, dim=1)
                                    action = int(
                                        prob.max(1)[1].data.cpu().numpy()[0])
                                    planner_actions.append(action)

                                pred_actions.append(action)
                                img, _, episode_done = h3d.step(action)

                                episode_done = episode_done or episode_length >= args.max_episode_length

                                img = torch.from_numpy(img.transpose(
                                    2, 0, 1)).float() / 255.0
                                img_feat_var = eval_loader.dataset.cnn(
                                    Variable(img.view(1, 3, 224, 224)
                                             .cuda())).view(1, 1, 3200)

                                dists_to_target.append(
                                    h3d.get_dist_to_target(h3d.env.cam.pos))
                                pos_queue.append([
                                    h3d.env.cam.pos.x, h3d.env.cam.pos.y,
                                    h3d.env.cam.pos.z, h3d.env.cam.yaw
                                ])

                                if episode_done == True:
                                    break

                                # query controller to continue or not
                                controller_action_in = Variable(
                                    torch.LongTensor(1,
                                                     1).fill_(action).cuda())
                                controller_scores = nav_model.controller_step(
                                    img_feat_var, controller_action_in,
                                    planner_hidden[0])

                                prob = F.softmax(controller_scores, dim=1)
                                controller_action = int(
                                    prob.max(1)[1].data.cpu().numpy()[0])

                                if controller_action == 1 and controller_action_counter < 4:
                                    controller_action_counter += 1
                                    controller_step = True
                                else:
                                    controller_action_counter = 0
                                    controller_step = False
                                    controller_action = 0

                                controller_actions.append(controller_action)

                                action_in = torch.LongTensor(
                                    1, 1).fill_(action + 1).cuda()

                        # run answerer here
                        if len(pos_queue) < 5:
                            pos_queue = eval_loader.dataset.episode_pos_queue[len(
                                pos_queue) - 5:] + pos_queue
                        images = eval_loader.dataset.get_frames(
                            h3d, pos_queue[-5:], preprocess=True)
                        images_var = Variable(
                            torch.from_numpy(images).cuda()).view(
                                1, 5, 3, 224, 224)
                        scores, att_probs = ans_model(images_var, question_var)
                        ans_acc, ans_rank = vqa_metrics.compute_ranks(
                            scores.data.cpu(), answer)

                        pred_answer = scores.max(1)[1].data[0]

                        print('[Q_GT]', ' '.join([
                            eval_loader.dataset.vocab['questionIdxToToken'][x]
                            for x in question[0] if x != 0
                        ]))
                        print('[A_GT]', eval_loader.dataset.vocab[
                            'answerIdxToToken'][answer[0]])
                        print('[A_PRED]', eval_loader.dataset.vocab[
                            'answerIdxToToken'][pred_answer])

                        # compute stats
                        metrics_slug['accuracy_' + str(i)] = ans_acc[0]
                        metrics_slug['mean_rank_' + str(i)] = ans_rank[0]
                        metrics_slug['mean_reciprocal_rank_'
                                     + str(i)] = 1.0 / ans_rank[0]

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

                    # navigation metrics
                    metrics_list = []
                    for i in nav_metrics.metric_names:
                        if i not in metrics_slug:
                            metrics_list.append(nav_metrics.metrics[
                                nav_metrics.metric_names.index(i)][0])
                        else:
                            metrics_list.append(metrics_slug[i])

                    nav_metrics.update(metrics_list)

                    # vqa metrics
                    metrics_list = []
                    for i in vqa_metrics.metric_names:
                        if i not in metrics_slug:
                            metrics_list.append(vqa_metrics.metrics[
                                vqa_metrics.metric_names.index(i)][0])
                        else:
                            metrics_list.append(metrics_slug[i])

                    vqa_metrics.update(metrics_list)

                try:
                    print(nav_metrics.get_stat_string(mode=0))
                    print(vqa_metrics.get_stat_string(mode=0))
                except:
                    pass

                print('epoch', epoch)
                print('invalids', len(invalids))

                eval_loader.dataset._load_envs()
                if len(eval_loader.dataset.pruned_env_set) == 0:
                    done = True

        epoch += 1

        # checkpoint if best val accuracy
        if vqa_metrics.metrics[2][0] > best_eval_acc:  # ans_acc_50
            best_eval_acc = vqa_metrics.metrics[2][0]
            if epoch % args.eval_every == 0 and args.log == True:
                vqa_metrics.dump_log()
                nav_metrics.dump_log()

                model_state = get_state(nav_model)

                aad = dict(args.__dict__)
                ad = {}
                for i in aad:
                    if i[0] != '_':
                        ad[i] = aad[i]

                checkpoint = {'args': ad, 'state': model_state, 'epoch': epoch}

                checkpoint_path = '%s/epoch_%d_ans_50_%.04f.pt' % (
                    args.checkpoint_dir, epoch, best_eval_acc)
                print('Saving checkpoint to %s' % checkpoint_path)
                torch.save(checkpoint, checkpoint_path)

        print('[best_eval_ans_acc_50:%.04f]' % best_eval_acc)

        eval_loader.dataset._load_envs(start_idx=0, in_order=True)


def train(rank, args, shared_nav_model, shared_ans_model):

    torch.cuda.set_device(args.gpus.index(args.gpus[rank % len(args.gpus)]))

    if args.model_type == 'pacman':

        model_kwargs = {'question_vocab': load_vocab(args.vocab_json)}
        nav_model = NavPlannerControllerModel(**model_kwargs)

    else:

        exit()

    model_kwargs = {'vocab': load_vocab(args.vocab_json)}
    ans_model = VqaLstmCnnAttentionModel(**model_kwargs)

    optim = torch.optim.SGD(
        filter(lambda p: p.requires_grad, shared_nav_model.parameters()),
        lr=args.learning_rate)

    train_loader_kwargs = {
        'questions_h5': args.train_h5,
        'data_json': args.data_json,
        'vocab': args.vocab_json,
        'target_obj_conn_map_dir': args.target_obj_conn_map_dir,
        'map_resolution': args.map_resolution,
        'batch_size': 1,
        'input_type': args.model_type,
        'num_frames': 5,
        'split': 'train',
        'max_threads_per_gpu': args.max_threads_per_gpu,
        'gpu_id': args.gpus[rank % len(args.gpus)],
        'to_cache': args.cache,
        'max_controller_actions': args.max_controller_actions,
        'max_actions': args.max_actions
    }

    args.output_nav_log_path = os.path.join(args.log_dir,
                                            'nav_train_' + str(rank) + '.json')
    args.output_ans_log_path = os.path.join(args.log_dir,
                                            'ans_train_' + str(rank) + '.json')

    nav_model.load_state_dict(shared_nav_model.state_dict())
    nav_model.cuda()

    ans_model.load_state_dict(shared_ans_model.state_dict())
    ans_model.eval()
    ans_model.cuda()

    nav_metrics = NavMetric(
        info={'split': 'train',
              'thread': rank},
        metric_names=[
            'planner_loss', 'controller_loss', 'reward', 'episode_length'
        ],
        log_json=args.output_nav_log_path)

    vqa_metrics = VqaMetric(
        info={'split': 'train',
              'thread': rank},
        metric_names=['accuracy', 'mean_rank', 'mean_reciprocal_rank'],
        log_json=args.output_ans_log_path)

    train_loader = EqaDataLoader(**train_loader_kwargs)

    print('train_loader has %d samples' % len(train_loader.dataset))

    t, epoch = 0, 0
    p_losses, c_losses, reward_list, episode_length_list = [], [], [], []

    nav_metrics.update([10.0, 10.0, 0, 100])

    mult = 0.1

    while epoch < int(args.max_epochs):

        if 'pacman' in args.model_type:

            planner_lossFn = MaskedNLLCriterion().cuda()
            controller_lossFn = MaskedNLLCriterion().cuda()

            done = False
            all_envs_loaded = train_loader.dataset._check_if_all_envs_loaded()

            while done == False:

                for batch in train_loader:

                    nav_model.load_state_dict(shared_nav_model.state_dict())
                    nav_model.eval()
                    nav_model.cuda()

                    idx, question, answer, actions, action_length = batch
                    metrics_slug = {}

                    h3d = train_loader.dataset.episode_house

                    # evaluate at multiple initializations
                    # for i in [10, 30, 50]:

                    t += 1

                    question_var = Variable(question.cuda())

                    controller_step = False
                    planner_hidden = nav_model.planner_nav_rnn.init_hidden(1)

                    # forward through planner till spawn
                    (
                        planner_actions_in, planner_img_feats,
                        controller_step, controller_action_in,
                        controller_img_feat, init_pos,
                        controller_action_counter
                    ) = train_loader.dataset.get_hierarchical_features_till_spawn(
                        actions[0, :action_length[0] + 1].numpy(), max(3, int(mult * action_length[0])), args.max_controller_actions
                    )

                    planner_actions_in_var = Variable(
                        planner_actions_in.cuda())
                    planner_img_feats_var = Variable(planner_img_feats.cuda())

                    for step in range(planner_actions_in.size(0)):

                        planner_scores, planner_hidden = nav_model.planner_step(
                            question_var, planner_img_feats_var[step].view(
                                1, 1, 3200), planner_actions_in_var[step].view(
                                    1, 1), planner_hidden)

                    if controller_step == True:

                        controller_img_feat_var = Variable(
                            controller_img_feat.cuda())
                        controller_action_in_var = Variable(
                            torch.LongTensor(1, 1).fill_(
                                int(controller_action_in)).cuda())

                        controller_scores = nav_model.controller_step(
                            controller_img_feat_var.view(1, 1, 3200),
                            controller_action_in_var.view(1, 1),
                            planner_hidden[0])

                        prob = F.softmax(controller_scores, dim=1)
                        controller_action = int(
                            prob.max(1)[1].data.cpu().numpy()[0])

                        if controller_action == 1:
                            controller_step = True
                        else:
                            controller_step = False

                        action = int(controller_action_in)
                        action_in = torch.LongTensor(
                            1, 1).fill_(action + 1).cuda()

                    else:

                        prob = F.softmax(planner_scores, dim=1)
                        action = int(prob.max(1)[1].data.cpu().numpy()[0])

                        action_in = torch.LongTensor(
                            1, 1).fill_(action + 1).cuda()

                    h3d.env.reset(
                        x=init_pos[0], y=init_pos[2], yaw=init_pos[3])

                    init_dist_to_target = h3d.get_dist_to_target(
                        h3d.env.cam.pos)
                    if init_dist_to_target < 0:  # unreachable
                        # invalids.append([idx[0], i])
                        continue

                    episode_length = 0
                    episode_done = True
                    controller_action_counter = 0

                    dists_to_target, pos_queue = [init_dist_to_target], [
                        init_pos
                    ]

                    rewards, planner_actions, planner_log_probs, controller_actions, controller_log_probs = [], [], [], [], []

                    if action != 3:

                        # take the first step
                        img, rwd, episode_done = h3d.step(action, step_reward=True)
                        img = torch.from_numpy(img.transpose(
                            2, 0, 1)).float() / 255.0
                        img_feat_var = train_loader.dataset.cnn(
                            Variable(img.view(1, 3, 224, 224).cuda())).view(
                                1, 1, 3200)

                        for step in range(args.max_episode_length):

                            episode_length += 1

                            if controller_step == False:
                                planner_scores, planner_hidden = nav_model.planner_step(
                                    question_var, img_feat_var,
                                    Variable(action_in), planner_hidden)

                                planner_prob = F.softmax(planner_scores, dim=1)
                                planner_log_prob = F.log_softmax(
                                    planner_scores, dim=1)

                                action = planner_prob.multinomial().data
                                planner_log_prob = planner_log_prob.gather(
                                    1, Variable(action))

                                planner_log_probs.append(
                                    planner_log_prob.cpu())

                                action = int(action.cpu().numpy()[0, 0])
                                planner_actions.append(action)

                            img, rwd, episode_done = h3d.step(action, step_reward=True)

                            episode_done = episode_done or episode_length >= args.max_episode_length

                            rewards.append(rwd)

                            img = torch.from_numpy(img.transpose(
                                2, 0, 1)).float() / 255.0
                            img_feat_var = train_loader.dataset.cnn(
                                Variable(img.view(1, 3, 224, 224)
                                         .cuda())).view(1, 1, 3200)

                            dists_to_target.append(
                                h3d.get_dist_to_target(h3d.env.cam.pos))
                            pos_queue.append([
                                h3d.env.cam.pos.x, h3d.env.cam.pos.y,
                                h3d.env.cam.pos.z, h3d.env.cam.yaw
                            ])

                            if episode_done == True:
                                break

                            # query controller to continue or not
                            controller_action_in = Variable(
                                torch.LongTensor(1, 1).fill_(action).cuda())
                            controller_scores = nav_model.controller_step(
                                img_feat_var, controller_action_in,
                                planner_hidden[0])

                            controller_prob = F.softmax(
                                controller_scores, dim=1)
                            controller_log_prob = F.log_softmax(
                                controller_scores, dim=1)

                            controller_action = controller_prob.multinomial(
                            ).data

                            if int(controller_action[0]
                                   ) == 1 and controller_action_counter < 4:
                                controller_action_counter += 1
                                controller_step = True
                            else:
                                controller_action_counter = 0
                                controller_step = False
                                controller_action.fill_(0)

                            controller_log_prob = controller_log_prob.gather(
                                1, Variable(controller_action))
                            controller_log_probs.append(
                                controller_log_prob.cpu())

                            controller_action = int(
                                controller_action.cpu().numpy()[0, 0])
                            controller_actions.append(controller_action)
                            action_in = torch.LongTensor(
                                1, 1).fill_(action + 1).cuda()

                    # run answerer here
                    ans_acc = [0]
                    if action == 3:
                        if len(pos_queue) < 5:
                            pos_queue = train_loader.dataset.episode_pos_queue[len(
                                pos_queue) - 5:] + pos_queue
                        images = train_loader.dataset.get_frames(
                            h3d, pos_queue[-5:], preprocess=True)
                        images_var = Variable(
                            torch.from_numpy(images).cuda()).view(
                                1, 5, 3, 224, 224)
                        scores, att_probs = ans_model(images_var, question_var)
                        ans_acc, ans_rank = vqa_metrics.compute_ranks(
                            scores.data.cpu(), answer)
                        vqa_metrics.update([ans_acc, ans_rank, 1.0 / ans_rank])

                    rewards.append(h3d.success_reward * ans_acc[0])

                    R = torch.zeros(1, 1)

                    planner_loss = 0
                    controller_loss = 0

                    planner_rev_idx = -1
                    for i in reversed(range(len(rewards))):
                        R = 0.99 * R + rewards[i]
                        advantage = R - nav_metrics.metrics[2][1]

                        if i < len(controller_actions):
                            controller_loss = controller_loss - controller_log_probs[i] * Variable(
                                advantage)

                            if controller_actions[i] == 0 and planner_rev_idx + len(planner_log_probs) >= 0:
                                planner_loss = planner_loss - planner_log_probs[planner_rev_idx] * Variable(
                                    advantage)
                                planner_rev_idx -= 1

                        elif planner_rev_idx + len(planner_log_probs) >= 0:

                            planner_loss = planner_loss - planner_log_probs[planner_rev_idx] * Variable(
                                advantage)
                            planner_rev_idx -= 1

                    controller_loss /= max(1, len(controller_log_probs))
                    planner_loss /= max(1, len(planner_log_probs))

                    optim.zero_grad()

                    if isinstance(planner_loss, float) == False and isinstance(
                            controller_loss, float) == False:
                        p_losses.append(planner_loss.data[0, 0])
                        c_losses.append(controller_loss.data[0, 0])
                        reward_list.append(np.sum(rewards))
                        episode_length_list.append(episode_length)

                        (planner_loss + controller_loss).backward()

                        ensure_shared_grads(nav_model.cpu(), shared_nav_model)
                        optim.step()

                    if len(reward_list) > 50:

                        nav_metrics.update([
                            p_losses, c_losses, reward_list,
                            episode_length_list
                        ])

                        print(nav_metrics.get_stat_string())
                        if args.log == True:
                            nav_metrics.dump_log()

                        if nav_metrics.metrics[2][1] > 0.35:
                            mult = min(mult + 0.1, 1.0)

                        p_losses, c_losses, reward_list, episode_length_list = [], [], [], []

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
        default='/path/to/target-obj-conn-maps/500')
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
        default='pacman',
        choices=['cnn', 'cnn+q', 'lstm', 'lstm+q', 'pacman'])
    parser.add_argument('-max_episode_length', default=100, type=int)

    # optim params
    parser.add_argument('-batch_size', default=20, type=int)
    parser.add_argument('-learning_rate', default=1e-5, type=float)
    parser.add_argument('-max_epochs', default=1000, type=int)

    # bookkeeping
    parser.add_argument('-print_every', default=5, type=int)
    parser.add_argument('-eval_every', default=1, type=int)
    parser.add_argument('-identifier', default='cnn')
    parser.add_argument('-num_processes', default=1, type=int)
    parser.add_argument('-max_threads_per_gpu', default=10, type=int)

    # checkpointing
    parser.add_argument('-nav_checkpoint_path', default=False)
    parser.add_argument('-ans_checkpoint_path', default=False)

    parser.add_argument('-checkpoint_dir', default='checkpoints/eqa/')
    parser.add_argument('-log_dir', default='logs/eqa/')
    parser.add_argument('-log', default=False, action='store_true')
    parser.add_argument('-cache', default=False, action='store_true')
    parser.add_argument('-max_controller_actions', type=int, default=5)
    parser.add_argument('-max_actions', type=int)
    args = parser.parse_args()

    args.time_id = time.strftime("%m_%d_%H:%M")

    try:
        args.gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        args.gpus = [int(x) for x in args.gpus]
    except KeyError:
        print("CPU not supported")
        exit()

    # Load navigation model
    if args.nav_checkpoint_path != False:
        print('Loading navigation checkpoint from %s' % args.nav_checkpoint_path)
        checkpoint = torch.load(
            args.nav_checkpoint_path, map_location={
                'cuda:0': 'cpu'
            })

        args_to_keep = ['model_type']

        for i in args.__dict__:
            if i not in args_to_keep:
                checkpoint['args'][i] = args.__dict__[i]

        args = type('new_dict', (object, ), checkpoint['args'])

    args.checkpoint_dir = os.path.join(args.checkpoint_dir,
                                       args.time_id + '_' + args.identifier)
    args.log_dir = os.path.join(args.log_dir,
                                args.time_id + '_' + args.identifier)
    print(args.__dict__)

    if not os.path.exists(args.checkpoint_dir) and args.log == True:
        os.makedirs(args.checkpoint_dir)
        os.makedirs(args.log_dir)

    if args.model_type == 'pacman':

        model_kwargs = {'question_vocab': load_vocab(args.vocab_json)}
        shared_nav_model = NavPlannerControllerModel(**model_kwargs)

    else:

        exit()

    shared_nav_model.share_memory()

    if args.nav_checkpoint_path != False:
        print('Loading navigation params from checkpoint: %s' %
            args.nav_checkpoint_path)
        shared_nav_model.load_state_dict(checkpoint['state'])

    # Load answering model
    if args.ans_checkpoint_path != False:
        print('Loading answering checkpoint from %s' % args.ans_checkpoint_path)
        ans_checkpoint = torch.load(
            args.ans_checkpoint_path, map_location={
                'cuda:0': 'cpu'
            })

    ans_model_kwargs = {'vocab': load_vocab(args.vocab_json)}
    shared_ans_model = VqaLstmCnnAttentionModel(**ans_model_kwargs)

    shared_ans_model.share_memory()

    if args.ans_checkpoint_path != False:
        print('Loading params from checkpoint: %s' % args.ans_checkpoint_path)
        shared_ans_model.load_state_dict(ans_checkpoint['state'])

    if args.mode == 'eval':

        eval(0, args, shared_nav_model, shared_ans_model)

    elif args.mode == 'train':

        train(0, args, shared_nav_model, shared_ans_model)

    else:

        processes = []

        p = mp.Process(
            target=eval, args=(0, args, shared_nav_model, shared_ans_model))
        p.start()
        processes.append(p)

        for rank in range(1, args.num_processes + 1):
            p = mp.Process(
                target=train,
                args=(rank, args, shared_nav_model, shared_ans_model))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()