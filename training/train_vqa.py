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

import torch
from torch.autograd import Variable
torch.backends.cudnn.enabled = False
import torch.multiprocessing as mp

from models import VqaLstmModel, VqaLstmCnnAttentionModel
from data import EqaDataset, EqaDataLoader
from metrics import VqaMetric

from models import get_state, repackage_hidden, ensure_shared_grads
from data import load_vocab

import pdb


def eval(rank, args, shared_model):

    torch.cuda.set_device(args.gpus.index(args.gpus[rank % len(args.gpus)]))

    if args.input_type == 'ques':

        model_kwargs = {'vocab': load_vocab(args.vocab_json)}
        model = VqaLstmModel(**model_kwargs)

    elif args.input_type == 'ques,image':

        model_kwargs = {'vocab': load_vocab(args.vocab_json)}
        model = VqaLstmCnnAttentionModel(**model_kwargs)

    lossFn = torch.nn.CrossEntropyLoss().cuda()

    eval_loader_kwargs = {
        'questions_h5': getattr(args, args.eval_split + '_h5'),
        'data_json': args.data_json,
        'vocab': args.vocab_json,
        'batch_size': 1,
        'input_type': args.input_type,
        'num_frames': args.num_frames,
        'split': args.eval_split,
        'max_threads_per_gpu': args.max_threads_per_gpu,
        'gpu_id': args.gpus[rank%len(args.gpus)],
        'to_cache': args.cache
    }

    eval_loader = EqaDataLoader(**eval_loader_kwargs)
    print('eval_loader has %d samples' % len(eval_loader.dataset))

    args.output_log_path = os.path.join(args.log_dir,
                                        'eval_' + str(rank) + '.json')

    t, epoch, best_eval_acc = 0, 0, 0

    while epoch < int(args.max_epochs):

        model.load_state_dict(shared_model.state_dict())
        model.eval()

        metrics = VqaMetric(
            info={'split': args.eval_split},
            metric_names=[
                'loss', 'accuracy', 'mean_rank', 'mean_reciprocal_rank'
            ],
            log_json=args.output_log_path)

        if args.input_type == 'ques':
            for batch in eval_loader:
                t += 1

                model.cuda()

                idx, questions, answers = batch

                questions_var = Variable(questions.cuda())
                answers_var = Variable(answers.cuda())

                scores = model(questions_var)
                loss = lossFn(scores, answers_var)

                # update metrics
                accuracy, ranks = metrics.compute_ranks(
                    scores.data.cpu(), answers)
                metrics.update([loss.data[0], accuracy, ranks, 1.0 / ranks])

            print(metrics.get_stat_string(mode=0))

        elif args.input_type == 'ques,image':
            done = False
            all_envs_loaded = eval_loader.dataset._check_if_all_envs_loaded()

            while done == False:
                for batch in eval_loader:
                    t += 1

                    model.cuda()

                    idx, questions, answers, images, _, _, _ = batch

                    questions_var = Variable(questions.cuda())
                    answers_var = Variable(answers.cuda())
                    images_var = Variable(images.cuda())

                    scores, att_probs = model(images_var, questions_var)
                    loss = lossFn(scores, answers_var)

                    # update metrics
                    accuracy, ranks = metrics.compute_ranks(
                        scores.data.cpu(), answers)
                    metrics.update(
                        [loss.data[0], accuracy, ranks, 1.0 / ranks])

                print(metrics.get_stat_string(mode=0))

                if all_envs_loaded == False:
                    eval_loader.dataset._load_envs()
                    if len(eval_loader.dataset.pruned_env_set) == 0:
                        done = True
                else:
                    done = True

        epoch += 1

        # checkpoint if best val accuracy
        if metrics.metrics[1][0] > best_eval_acc:
            best_eval_acc = metrics.metrics[1][0]
            if epoch % args.eval_every == 0 and args.log == True:
                metrics.dump_log()

                model_state = get_state(model)

                if args.checkpoint_path != False:
                    ad = checkpoint['args']
                else:
                    ad = args.__dict__

                checkpoint = {'args': ad, 'state': model_state, 'epoch': epoch}

                checkpoint_path = '%s/epoch_%d_accuracy_%.04f.pt' % (
                    args.checkpoint_dir, epoch, best_eval_acc)
                print('Saving checkpoint to %s' % checkpoint_path)
                torch.save(checkpoint, checkpoint_path)

        print('[best_eval_accuracy:%.04f]' % best_eval_acc)


def train(rank, args, shared_model):

    torch.cuda.set_device(args.gpus.index(args.gpus[rank % len(args.gpus)]))

    if args.input_type == 'ques':

        model_kwargs = {'vocab': load_vocab(args.vocab_json)}
        model = VqaLstmModel(**model_kwargs)

    elif args.input_type == 'ques,image':

        model_kwargs = {'vocab': load_vocab(args.vocab_json)}
        model = VqaLstmCnnAttentionModel(**model_kwargs)

    lossFn = torch.nn.CrossEntropyLoss().cuda()

    optim = torch.optim.Adam(
        filter(lambda p: p.requires_grad, shared_model.parameters()),
        lr=args.learning_rate)

    train_loader_kwargs = {
        'questions_h5': args.train_h5,
        'data_json': args.data_json,
        'vocab': args.vocab_json,
        'batch_size': args.batch_size,
        'input_type': args.input_type,
        'num_frames': args.num_frames,
        'split': 'train',
        'max_threads_per_gpu': args.max_threads_per_gpu,
        'gpu_id': args.gpus[rank%len(args.gpus)],
        'to_cache': args.cache
    }

    args.output_log_path = os.path.join(args.log_dir,
                                        'train_' + str(rank) + '.json')

    metrics = VqaMetric(
        info={'split': 'train',
              'thread': rank},
        metric_names=['loss', 'accuracy', 'mean_rank', 'mean_reciprocal_rank'],
        log_json=args.output_log_path)

    train_loader = EqaDataLoader(**train_loader_kwargs)
    if args.input_type == 'ques,image':
        train_loader.dataset._load_envs(start_idx=0, in_order=True)

    print('train_loader has %d samples' % len(train_loader.dataset))

    t, epoch = 0, 0

    while epoch < int(args.max_epochs):

        if args.input_type == 'ques':

            for batch in train_loader:

                t += 1

                model.load_state_dict(shared_model.state_dict())
                model.train()
                model.cuda()

                idx, questions, answers = batch

                questions_var = Variable(questions.cuda())
                answers_var = Variable(answers.cuda())

                scores = model(questions_var)
                loss = lossFn(scores, answers_var)

                # zero grad
                optim.zero_grad()

                # update metrics
                accuracy, ranks = metrics.compute_ranks(scores.data.cpu(), answers)
                metrics.update([loss.data[0], accuracy, ranks, 1.0 / ranks])

                # backprop and update
                loss.backward()

                ensure_shared_grads(model.cpu(), shared_model)
                optim.step()

                if t % args.print_every == 0:
                    print(metrics.get_stat_string())
                    if args.log == True:
                        metrics.dump_log()

        elif args.input_type == 'ques,image':

            done = False
            all_envs_loaded = train_loader.dataset._check_if_all_envs_loaded()

            while done == False:

                for batch in train_loader:

                    t += 1

                    model.load_state_dict(shared_model.state_dict())
                    model.train()
                    model.cnn.eval()
                    model.cuda()

                    idx, questions, answers, images, _, _, _ = batch

                    questions_var = Variable(questions.cuda())
                    answers_var = Variable(answers.cuda())
                    images_var = Variable(images.cuda())

                    scores, att_probs = model(images_var, questions_var)
                    loss = lossFn(scores, answers_var)

                    # zero grad
                    optim.zero_grad()

                    # update metrics
                    accuracy, ranks = metrics.compute_ranks(scores.data.cpu(), answers)
                    metrics.update([loss.data[0], accuracy, ranks, 1.0 / ranks])

                    # backprop and update
                    loss.backward()

                    ensure_shared_grads(model.cpu(), shared_model)
                    optim.step()

                    if t % args.print_every == 0:
                        print(metrics.get_stat_string())
                        if args.log == True:
                            metrics.dump_log()

                if all_envs_loaded == False:
                    print('[CHECK][Cache:%d][Total:%d]' % (len(train_loader.dataset.img_data_cache),
                        len(train_loader.dataset.env_list)))
                    train_loader.dataset._load_envs(in_order=True)
                    if len(train_loader.dataset.pruned_env_set) == 0:
                        done = True
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

    parser.add_argument('-train_cache_path', default=False)
    parser.add_argument('-val_cache_path', default=False)

    parser.add_argument('-mode', default='train', type=str, choices=['train','eval'])
    parser.add_argument('-eval_split', default='val', type=str)

    # model details
    parser.add_argument(
        '-input_type', default='ques,image', choices=['ques', 'ques,image'])
    parser.add_argument(
        '-num_frames', default=5,
        type=int)  # -1 = all frames of navigation sequence

    # optim params
    parser.add_argument('-batch_size', default=20, type=int)
    parser.add_argument('-learning_rate', default=3e-4, type=float)
    parser.add_argument('-max_epochs', default=1000, type=int)

    # bookkeeping
    parser.add_argument('-print_every', default=50, type=int)
    parser.add_argument('-eval_every', default=1, type=int)
    parser.add_argument('-identifier', default='q-only')
    parser.add_argument('-num_processes', default=1, type=int)
    parser.add_argument('-max_threads_per_gpu', default=10, type=int)

    # checkpointing
    parser.add_argument('-checkpoint_path', default=False)
    parser.add_argument('-checkpoint_dir', default='checkpoints/vqa/')
    parser.add_argument('-log_dir', default='logs/vqa/')
    parser.add_argument('-log', default=False, action='store_true')
    parser.add_argument('-cache', default=False, action='store_true')
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

        args_to_keep = ['input_type', 'num_frames']

        checkpoint = torch.load(args.checkpoint_path, map_location={'cuda:0': 'cpu'})

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

    if args.input_type == 'ques':

        model_kwargs = {'vocab': load_vocab(args.vocab_json)}
        shared_model = VqaLstmModel(**model_kwargs)

    elif args.input_type == 'ques,image':

        model_kwargs = {'vocab': load_vocab(args.vocab_json)}
        shared_model = VqaLstmCnnAttentionModel(**model_kwargs)

    if args.checkpoint_path != False:
        print('Loading params from checkpoint: %s' % args.checkpoint_path)
        shared_model.load_state_dict(checkpoint['state'])

    shared_model.share_memory()

    if args.mode == 'eval':

        eval(0, args, shared_model)

    else:

        processes = []

        # Start the eval thread
        p = mp.Process(target=eval, args=(0, args, shared_model))
        p.start()
        processes.append(p)

        # Start the training thread(s)
        for rank in range(1, args.num_processes + 1):
            p = mp.Process(target=train, args=(rank, args, shared_model))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
