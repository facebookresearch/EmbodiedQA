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

from models import NavCnnModel, NavCnnRnnModel, NavPlannerControllerModel
from data import EqaDataset, EqaDataLoader
from metrics import NavMetric

from models import MaskedNLLCriterion

from models import get_state, repackage_hidden, ensure_shared_grads
from data import load_vocab, flat_to_hierarchical_actions

def eval(rank, args, shared_model):

    torch.cuda.set_device(args.gpus.index(args.gpus[rank % len(args.gpus)]))

    if args.model_type == 'pacman':

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
        'to_cache': False
    }

    eval_loader = EqaDataLoader(**eval_loader_kwargs)
    print('eval_loader has %d samples' % len(eval_loader.dataset))

    args.output_log_path = os.path.join(args.log_dir,
                                        'eval_' + str(rank) + '.json')

    t, epoch, best_eval_acc = 0, 0, 0.0

    while epoch < int(args.max_epochs):

        start_time = time.time()
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

        if 'pacman' in args.model_type:

            done = False

            while done == False:

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

                        # forward through planner till spawn
                        planner_actions_in, planner_img_feats, controller_step, controller_action_in, controller_img_feat, init_pos = eval_loader.dataset.get_hierarchical_features_till_spawn(
                            actions[0, :action_length[0] + 1].numpy(), i)

                        planner_actions_in_var = Variable(
                            planner_actions_in.cuda())
                        planner_img_feats_var = Variable(
                            planner_img_feats.cuda())

                        for step in range(planner_actions_in.size(0)):

                            planner_scores, planner_hidden = model.planner_step(
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

                            controller_scores = model.controller_step(
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
                                    planner_scores, planner_hidden = model.planner_step(
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
                                controller_scores = model.controller_step(
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
                except:
                    pass

                print('epoch', epoch)
                print('invalids', len(invalids))

                eval_loader.dataset._load_envs()
                if len(eval_loader.dataset.pruned_env_set) == 0:
                    done = True

        epoch += 1

        # checkpoint if best val loss
        if metrics.metrics[8][0] > best_eval_acc:  # d_D_50
            best_eval_acc = metrics.metrics[8][0]
            if epoch % args.eval_every == 0 and args.to_log == 1:
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
                torch.save(checkpoint, checkpoint_path)

        print('[best_eval_d_D_50:%.04f]' % best_eval_acc)

        eval_loader.dataset._load_envs(start_idx=0, in_order=True)


def train(rank, args, shared_model):
    torch.cuda.set_device(args.gpus.index(args.gpus[rank % len(args.gpus)]))

    if args.model_type == 'pacman':

        model_kwargs = {'question_vocab': load_vocab(args.vocab_json)}
        model = NavPlannerControllerModel(**model_kwargs)

    else:

        exit()

    lossFn = torch.nn.CrossEntropyLoss().cuda()

    optim = torch.optim.Adam(
        filter(lambda p: p.requires_grad, shared_model.parameters()),
        lr=args.learning_rate)

    train_loader_kwargs = {
        'questions_h5': args.train_h5,
        'data_json': args.data_json,
        'vocab': args.vocab_json,
        'batch_size': args.batch_size,
        'input_type': args.model_type,
        'num_frames': 5,
        'split': 'train',
        'max_threads_per_gpu': args.max_threads_per_gpu,
        'gpu_id': args.gpus[rank % len(args.gpus)],
        'to_cache': args.to_cache
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

    t, epoch = 0, 0

    while epoch < int(args.max_epochs):

        if 'pacman' in args.model_type:

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

                    # backprop and update
                    (planner_loss + controller_loss).backward()

                    ensure_shared_grads(model.cpu(), shared_model)
                    optim.step()

                    if t % args.print_every == 0:
                        print(metrics.get_stat_string())
                        if args.to_log == 1:
                            metrics.dump_log()

                    print('[CHECK][Cache:%d][Total:%d]' %
                          (len(train_loader.dataset.img_data_cache),
                           len(train_loader.dataset.env_list)))

                if all_envs_loaded == False:
                    train_loader.dataset._load_envs(in_order=True)
                    if len(train_loader.dataset.pruned_env_set) == 0:
                        done = True
                        if args.to_cache == False:
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
        choices=['pacman'])
    parser.add_argument('-max_episode_length', default=100, type=int)

    # optim params
    parser.add_argument('-batch_size', default=20, type=int)
    parser.add_argument('-learning_rate', default=3e-4, type=float)
    parser.add_argument('-max_epochs', default=1000, type=int)

    # bookkeeping
    parser.add_argument('-print_every', default=5, type=int)
    parser.add_argument('-eval_every', default=1, type=int)
    parser.add_argument('-identifier', default='cnn')
    parser.add_argument('-num_processes', default=1, type=int)
    parser.add_argument('-max_threads_per_gpu', default=10, type=int)

    # checkpointing
    parser.add_argument('-checkpoint_path', default=False)
    parser.add_argument('-checkpoint_dir', default='checkpoints/nav/')
    parser.add_argument('-log_dir', default='logs/nav/')
    parser.add_argument('-to_log', default=0, type=int)
    parser.add_argument('-to_cache', action='store_true')
    args = parser.parse_args()

    args.time_id = time.strftime("%m_%d_%H:%M")

    try:
        args.gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        args.gpus = [int(x) for x in args.gpus]
    except KeyError:
        print("CPU not supported")
        exit()

    if args.checkpoint_path != False:

        print('Loading checkpoint from %s' % args.checkpoint_path)

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

    print(args.__dict__)

    if not os.path.exists(args.checkpoint_dir) and args.to_log == 1:
        os.makedirs(args.checkpoint_dir)
        os.makedirs(args.log_dir)

    if args.model_type == 'pacman':

        model_kwargs = {'question_vocab': load_vocab(args.vocab_json)}
        shared_model = NavPlannerControllerModel(**model_kwargs)

    else:

        exit()

    shared_model.share_memory()

    if args.checkpoint_path != False:
        print('Loading params from checkpoint: %s' % args.checkpoint_path)
        shared_model.load_state_dict(checkpoint['state'])

    if args.mode == 'eval':

        eval(0, args, shared_model)

    elif args.mode == 'train':

        train(0, args, shared_model)

    else:

        processes = []

        p = mp.Process(target=eval, args=(0, args, shared_model))
        p.start()
        processes.append(p)

        for rank in range(1, args.num_processes + 1):
            p = mp.Process(target=train, args=(rank, args, shared_model))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()