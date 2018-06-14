# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import time
import h5py
import argparse
import numpy as np
import os, sys, json
from tqdm import tqdm

from scipy.misc import imread, imresize

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.autograd import Variable

from House3D import objrender, Environment, load_config
from House3D.core import local_create_house

sys.path.insert(0, '../utils/')
from house3d import House3DUtils

from models import MultitaskCNN

import pdb


def load_vocab(path):
    with open(path, 'r') as f:
        vocab = json.load(f)
        vocab['questionIdxToToken'] = invert_dict(vocab['questionTokenToIdx'])
        vocab['answerIdxToToken'] = invert_dict(vocab['answerTokenToIdx'])

    assert vocab['questionTokenToIdx']['<NULL>'] == 0
    assert vocab['questionTokenToIdx']['<START>'] == 1
    assert vocab['questionTokenToIdx']['<END>'] == 2
    return vocab


def invert_dict(d):
    return {v: k for k, v in d.items()}


"""
if the action sequence is [f, f, l, l, f, f, f, r]

input sequence to planner is [<start>, f, l, f, r]
output sequence for planner is [f, l, f, r, <end>]

input sequences to controller are [f, f, l, l, f, f, f, r]
output sequences for controller are [1, 0, 1, 0, 1, 1, 0, 0]
"""
def flat_to_hierarchical_actions(actions=[]):
    assert len(actions) != 0

    controller_action_ctr, controller_action_lim = 0, 5

    planner_actions, controller_actions = [1], []
    prev_action = 1

    pq_idx, cq_idx, ph_idx = [], [], []
    ph_trck = 0

    for i in range(1, len(actions)):

        if actions[i] != prev_action:
            planner_actions.append(actions[i])
            pq_idx.append(i-1)

        if i > 1:
            ph_idx.append(ph_trck)
            if actions[i] == prev_action:
                controller_actions.append(1)
                controller_action_ctr += 1
            else:
                controller_actions.append(0)
                controller_action_ctr = 0
                ph_trck += 1
            cq_idx.append(i-1)


        prev_action = actions[i]

        if controller_action_ctr == controller_action_lim-1:
            prev_action = False

    return planner_actions, controller_actions, pq_idx, cq_idx, ph_idx


def _dataset_to_tensor(dset, mask=None, dtype=np.int64):
    arr = np.asarray(dset, dtype=dtype)
    if mask is not None:
        arr = arr[mask]
    if dtype == np.float32:
        tensor = torch.FloatTensor(arr)
    else:
        tensor = torch.LongTensor(arr)
    return tensor


def eqaCollateCnn(batch):
    transposed = list(zip(*batch))
    idx_batch = default_collate(transposed[0])
    question_batch = default_collate(transposed[1])
    answer_batch = default_collate(transposed[2])
    images_batch = default_collate(transposed[3])
    actions_in_batch = default_collate(transposed[4])
    actions_out_batch = default_collate(transposed[5])
    action_lengths_batch = default_collate(transposed[6])
    return [
        idx_batch, question_batch, answer_batch, images_batch,
        actions_in_batch, actions_out_batch, action_lengths_batch
    ]


def eqaCollateSeq2seq(batch):
    transposed = list(zip(*batch))
    idx_batch = default_collate(transposed[0])
    questions_batch = default_collate(transposed[1])
    answers_batch = default_collate(transposed[2])
    images_batch = default_collate(transposed[3])
    actions_in_batch = default_collate(transposed[4])
    actions_out_batch = default_collate(transposed[5])
    action_lengths_batch = default_collate(transposed[6])
    mask_batch = default_collate(transposed[7])

    return [
        idx_batch, questions_batch, answers_batch, images_batch,
        actions_in_batch, actions_out_batch, action_lengths_batch, mask_batch
    ]


class EqaDataset(Dataset):
    def __init__(self,
                 questions_h5,
                 vocab,
                 num_frames=1,
                 data_json=False,
                 split='train',
                 gpu_id=0,
                 input_type='ques',
                 max_threads_per_gpu=10,
                 to_cache=False,
                 target_obj_conn_map_dir=False,
                 map_resolution=1000):
        self.questions_h5 = questions_h5
        self.vocab = load_vocab(vocab)
        self.num_frames = num_frames

        np.random.seed()

        self.data_json = data_json
        self.split = split
        self.gpu_id = gpu_id

        self.input_type = input_type

        self.max_threads_per_gpu = max_threads_per_gpu

        self.target_obj_conn_map_dir = target_obj_conn_map_dir
        self.map_resolution = map_resolution

        self.to_cache = to_cache
        self.img_data_cache = {}

        if self.data_json != False:
            data = json.load(open(self.data_json, 'r'))
            self.envs = data['envs']

            self.env_idx = data[self.split + '_env_idx']
            self.env_list = [self.envs[x] for x in self.env_idx]
            self.env_set = list(set(self.env_list))
            self.env_set.sort()

            print('Total envs: %d' % len(list(set(self.envs))))
            print('Envs in %s: %d' % (self.split,
                                      len(list(set(self.env_idx)))))

            if input_type != 'ques':
                ''''
                If training, randomly sample and load a subset of environments,
                train on those, and then cycle through to load the rest.

                On the validation and test set, load in order, and cycle through.

                For both, add optional caching so that if all environments
                have been cycled through once, then no need to re-load and
                instead, just the cache can be used.
                '''

                self.api_threads = []
                self._load_envs(start_idx=0, in_order=True)

                cnn_kwargs = {'num_classes': 191, 'pretrained': True}
                self.cnn = MultitaskCNN(**cnn_kwargs)
                self.cnn.eval()
                self.cnn.cuda()

            self.pos_queue = data[self.split + '_pos_queue']
            self.boxes = data[self.split + '_boxes']

        print('Reading question data into memory')
        self.idx = _dataset_to_tensor(questions_h5['idx'])
        self.questions = _dataset_to_tensor(questions_h5['questions'])
        self.answers = _dataset_to_tensor(questions_h5['answers'])
        self.actions = _dataset_to_tensor(questions_h5['action_labels'])
        self.action_lengths = _dataset_to_tensor(
            questions_h5['action_lengths'])

        if input_type == 'pacman':

            self.planner_actions = self.actions.clone().fill_(0)
            self.controller_actions = self.actions.clone().fill_(-1)

            self.planner_action_lengths = self.action_lengths.clone().fill_(0)
            self.controller_action_lengths = self.action_lengths.clone().fill_(
                0)

            self.planner_hidden_idx = self.actions.clone().fill_(0)

            self.planner_pos_queue_idx, self.controller_pos_queue_idx = [], []

            # parsing flat actions to planner-controller hierarchy
            for i in tqdm(range(len(self.actions))):

                pa, ca, pq_idx, cq_idx, ph_idx = flat_to_hierarchical_actions(self.actions[i][:self.action_lengths[i]+1])

                self.planner_actions[i][:len(pa)] = torch.Tensor(pa)
                self.controller_actions[i][:len(ca)] = torch.Tensor(ca)

                self.planner_action_lengths[i] = len(pa)-1
                self.controller_action_lengths[i] = len(ca)

                self.planner_pos_queue_idx.append(pq_idx)
                self.controller_pos_queue_idx.append(cq_idx)

                self.planner_hidden_idx[i][:len(ca)] = torch.Tensor(ph_idx)

    def _pick_envs_to_load(self,
                           split='train',
                           max_envs=10,
                           start_idx=0,
                           in_order=False):
        if split in ['val', 'test'] or in_order == True:
            pruned_env_set = self.env_set[start_idx:start_idx + max_envs]
        else:
            if max_envs < len(self.env_set):
                env_inds = np.random.choice(
                    len(self.env_set), max_envs, replace=False)
            else:
                env_inds = np.random.choice(
                    len(self.env_set), max_envs, replace=True)
            pruned_env_set = [self.env_set[x] for x in env_inds]
        return pruned_env_set

    def _load_envs(self, start_idx=-1, in_order=False):
        if start_idx == -1:
            start_idx = self.env_set.index(self.pruned_env_set[-1]) + 1

        # Pick envs
        self.pruned_env_set = self._pick_envs_to_load(
            split=self.split,
            max_envs=self.max_threads_per_gpu,
            start_idx=start_idx,
            in_order=in_order)

        if len(self.pruned_env_set) == 0:
            return

        # Load api threads
        start = time.time()
        if len(self.api_threads) == 0:
            for i in range(len(self.pruned_env_set)):
                self.api_threads.append(
                    objrender.RenderAPIThread(
                        w=224, h=224, device=self.gpu_id))

        self.cfg = load_config('../House3D/tests/config.json')

        print('[%.02f] Loaded %d api threads' % (time.time() - start,
                                                 len(self.api_threads)))
        start = time.time()

        # Load houses
        from multiprocessing import Pool
        _args = ([h, self.cfg, self.map_resolution]
                 for h in self.pruned_env_set)
        with Pool(len(self.pruned_env_set)) as pool:
            self.all_houses = pool.starmap(local_create_house, _args)

        print('[%.02f] Loaded %d houses' % (time.time() - start,
                                            len(self.all_houses)))
        start = time.time()

        # Load envs
        self.env_loaded = {}
        for i in range(len(self.all_houses)):
            print('[%02d/%d][split:%s][gpu:%d][house:%s]' %
                  (i + 1, len(self.all_houses), self.split, self.gpu_id,
                   self.all_houses[i].house['id']))
            self.env_loaded[self.all_houses[i].house['id']] = House3DUtils(
                Environment(self.api_threads[i], self.all_houses[i], self.cfg),
                target_obj_conn_map_dir=self.target_obj_conn_map_dir,
                build_graph=False)

        # [TODO] Unused till now
        self.env_ptr = -1

        print('[%.02f] Loaded %d house3d envs' % (time.time() - start,
                                                  len(self.env_loaded)))

        # Mark available data indices
        self.available_idx = [
            i for i, v in enumerate(self.env_list) if v in self.env_loaded
        ]
        print('Available inds: %d' % len(self.available_idx))

        # Flag to check if loaded envs have been cycled through or not
        # [TODO] Unused till now
        self.all_envs_loaded = False

    def _clear_api_threads(self):
        for i in range(len(self.api_threads)):
            del self.api_threads[0]
        self.api_threads = []

    def _check_if_all_envs_loaded(self):
        print('[CHECK][Cache:%d][Total:%d]' % (len(self.img_data_cache),
                                               len(self.env_list)))
        if len(self.img_data_cache) == len(self.env_list):
            self.available_idx = [i for i, v in enumerate(self.env_list)]
            return True
        else:
            return False

    def set_camera(self, e, pos, robot_height=1.0):
        assert len(pos) == 4

        e.env.cam.pos.x = pos[0]
        e.env.cam.pos.y = robot_height
        e.env.cam.pos.z = pos[2]
        e.env.cam.yaw = pos[3]

        e.env.cam.updateDirection()

    def render(self, e):
        return e.env.render()

    def get_frames(self, e, pos_queue, preprocess=True):
        if isinstance(pos_queue, list) == False:
            pos_queue = [pos_queue]

        res = []
        for i in range(len(pos_queue)):
            self.set_camera(e, pos_queue[i])
            img = np.array(self.render(e), copy=False, dtype=np.float32)

            if preprocess == True:
                img = img.transpose(2, 0, 1)
                img = img / 255.0

            res.append(img)

        return np.array(res)

    def get_hierarchical_features_till_spawn(self, actions, backtrack_steps=0):

        action_length = len(actions)-1
        pa, ca, pq_idx, cq_idx, ph_idx = flat_to_hierarchical_actions(actions)

        target_pos_idx = action_length - backtrack_steps

        controller_step = True
        if target_pos_idx in pq_idx:
            controller_step = False

        pq_idx_pruned = [v for v in pq_idx if v <= target_pos_idx]
        pa_pruned = pa[:len(pq_idx_pruned)+1]

        images = self.get_frames(
            self.episode_house,
            self.episode_pos_queue,
            preprocess=True)
        raw_img_feats = self.cnn(
            Variable(torch.FloatTensor(images)
                     .cuda())).data.cpu().numpy().copy()

        controller_img_feat, controller_action_in = False, False
        if controller_step == True:
            controller_img_feat = torch.from_numpy(raw_img_feats[target_pos_idx].copy())
            controller_action_in = pa_pruned[-1] - 2

        planner_img_feats = torch.from_numpy(raw_img_feats[pq_idx_pruned].copy())
        planner_actions_in = torch.from_numpy(np.array(pa_pruned[:-1]) - 1)

        return planner_actions_in, planner_img_feats, controller_step, controller_action_in, controller_img_feat, self.episode_pos_queue[target_pos_idx]


    def __getitem__(self, index):
        # [VQA] question-only
        if self.input_type == 'ques':
            idx = self.idx[index]
            question = self.questions[index]
            answer = self.answers[index]

            return (idx, question, answer)

        # [VQA] question+image
        elif self.input_type == 'ques,image':
            index = self.available_idx[index]

            idx = self.idx[index]
            question = self.questions[index]
            answer = self.answers[index]

            action_length = self.action_lengths[index]
            actions = self.actions[index]

            actions_in = actions[action_length - self.num_frames:action_length]
            actions_out = actions[action_length - self.num_frames + 1:
                                  action_length + 1]

            if self.to_cache == True and index in self.img_data_cache:
                images = self.img_data_cache[index]
            else:
                pos_queue = self.pos_queue[index][
                    -self.num_frames:]  # last 5 frames
                images = self.get_frames(
                    self.env_loaded[self.env_list[index]],
                    pos_queue,
                    preprocess=True)
                if self.to_cache == True:
                    self.img_data_cache[index] = images.copy()

            return (idx, question, answer, images, actions_in, actions_out,
                    action_length)

        # [NAV] question+cnn
        elif self.input_type in ['cnn', 'cnn+q']:

            index = self.available_idx[index]

            idx = self.idx[index]
            question = self.questions[index]
            answer = self.answers[index]

            action_length = self.action_lengths[index]
            actions = self.actions[index]

            if self.to_cache == True and index in self.img_data_cache:
                img_feats = self.img_data_cache[index]
            else:
                pos_queue = self.pos_queue[index]
                images = self.get_frames(
                    self.env_loaded[self.env_list[index]],
                    pos_queue,
                    preprocess=True)
                img_feats = self.cnn(
                    Variable(torch.FloatTensor(images)
                             .cuda())).data.cpu().numpy().copy()
                if self.to_cache == True:
                    self.img_data_cache[index] = img_feats

            # for val or test (evaluation), or
            # when target_obj_conn_map_dir is defined (reinforce),
            # load entire shortest path navigation trajectory
            # and load connectivity map for intermediate rewards
            if self.split in ['val', 'test'
                              ] or self.target_obj_conn_map_dir != False:
                target_obj_id, target_room = False, False
                bbox_obj = [
                    x for x in self.boxes[index]
                    if x['type'] == 'object' and x['target'] == True
                ][0]['box']
                for obj_id in self.env_loaded[self.env_list[index]].objects:
                    box2 = self.env_loaded[self.env_list[index]].objects[
                        obj_id]['bbox']
                    if all([bbox_obj['min'][x] == box2['min'][x] for x in range(3)]) == True and \
                        all([bbox_obj['max'][x] == box2['max'][x] for x in range(3)]) == True:
                        target_obj_id = obj_id
                        break
                bbox_room = [
                    x for x in self.boxes[index]
                    if x['type'] == 'room' and x['target'] == False
                ][0]
                for room in self.env_loaded[self.env_list[
                        index]].env.house.all_rooms:
                    if all([room['bbox']['min'][i] == bbox_room['box']['min'][i] for i in range(3)]) and \
                        all([room['bbox']['max'][i] == bbox_room['box']['max'][i] for i in range(3)]):
                        target_room = room
                        break
                assert target_obj_id != False
                assert target_room != False
                self.env_loaded[self.env_list[index]].set_target_object(
                    self.env_loaded[self.env_list[index]].objects[
                        target_obj_id], target_room)

                # [NOTE] only works for batch size = 1
                self.episode_pos_queue = self.pos_queue[index]
                self.episode_house = self.env_loaded[self.env_list[index]]
                self.target_room = target_room
                self.target_obj = self.env_loaded[self.env_list[
                    index]].objects[target_obj_id]

                actions_in = actions[:action_length]
                actions_out = actions[1:action_length + 1] - 2

                return (idx, question, answer, img_feats, actions_in,
                        actions_out, action_length)

            # if action_length is n
            # images.shape[0] is also n
            # actions[0] is <START>
            # actions[n] is <END>

            # grab 5 random frames
            # [NOTE]: this'll break for longer-than-5 navigation sequences
            start_idx = np.random.choice(img_feats.shape[0] + 1 -
                                         self.num_frames)
            img_feats = img_feats[start_idx:start_idx + self.num_frames]

            actions_in = actions[start_idx:start_idx + self.num_frames]
            actions_out = actions[start_idx + self.num_frames] - 2

            return (idx, question, answer, img_feats, actions_in, actions_out,
                    action_length)

        # [NAV] question+lstm
        elif self.input_type in ['lstm', 'lstm+q']:

            index = self.available_idx[index]

            idx = self.idx[index]
            question = self.questions[index]
            answer = self.answers[index]

            action_length = self.action_lengths[index]
            actions = self.actions[index]

            if self.split == 'train':
                if self.to_cache == True and index in self.img_data_cache:
                    img_feats = self.img_data_cache[index]
                else:
                    pos_queue = self.pos_queue[index]
                    images = self.get_frames(
                        self.env_loaded[self.env_list[index]],
                        pos_queue,
                        preprocess=True)
                    raw_img_feats = self.cnn(
                        Variable(torch.FloatTensor(images)
                                 .cuda())).data.cpu().numpy().copy()
                    img_feats = np.zeros(
                        (self.actions.shape[1], raw_img_feats.shape[1]),
                        dtype=np.float32)
                    img_feats[:raw_img_feats.shape[
                        0], :] = raw_img_feats.copy()
                    if self.to_cache == True:
                        self.img_data_cache[index] = img_feats

            actions_in = actions.clone() - 1
            actions_out = actions[1:].clone() - 2

            actions_in[action_length:].fill_(0)
            mask = actions_out.clone().gt(-1)
            if len(actions_out) > action_length:
                actions_out[action_length:].fill_(0)

            # for val or test (evaluation), or
            # when target_obj_conn_map_dir is defined (reinforce),
            # load entire shortest path navigation trajectory
            # and load connectivity map for intermediate rewards
            if self.split in ['val', 'test'
                              ] or self.target_obj_conn_map_dir != False:
                target_obj_id, target_room = False, False
                bbox_obj = [
                    x for x in self.boxes[index]
                    if x['type'] == 'object' and x['target'] == True
                ][0]['box']
                for obj_id in self.env_loaded[self.env_list[index]].objects:
                    box2 = self.env_loaded[self.env_list[index]].objects[
                        obj_id]['bbox']
                    if all([bbox_obj['min'][x] == box2['min'][x] for x in range(3)]) == True and \
                        all([bbox_obj['max'][x] == box2['max'][x] for x in range(3)]) == True:
                        target_obj_id = obj_id
                        break
                bbox_room = [
                    x for x in self.boxes[index]
                    if x['type'] == 'room' and x['target'] == False
                ][0]
                for room in self.env_loaded[self.env_list[
                        index]].env.house.all_rooms:
                    if all([room['bbox']['min'][i] == bbox_room['box']['min'][i] for i in range(3)]) and \
                        all([room['bbox']['max'][i] == bbox_room['box']['max'][i] for i in range(3)]):
                        target_room = room
                        break
                assert target_obj_id != False
                assert target_room != False
                self.env_loaded[self.env_list[index]].set_target_object(
                    self.env_loaded[self.env_list[index]].objects[
                        target_obj_id], target_room)

                # [NOTE] only works for batch size = 1
                self.episode_pos_queue = self.pos_queue[index]
                self.episode_house = self.env_loaded[self.env_list[index]]
                self.target_room = target_room
                self.target_obj = self.env_loaded[self.env_list[
                    index]].objects[target_obj_id]

                return (idx, question, answer, False, actions_in, actions_out,
                        action_length, mask)

            return (idx, question, answer, img_feats, actions_in, actions_out,
                    action_length, mask)

        # [NAV] planner-controller
        elif self.input_type in ['pacman']:

            index = self.available_idx[index]

            idx = self.idx[index]
            question = self.questions[index]
            answer = self.answers[index]

            action_length = self.action_lengths[index]
            actions = self.actions[index]

            planner_actions = self.planner_actions[index]
            controller_actions = self.controller_actions[index]

            planner_action_length = self.planner_action_lengths[index]
            controller_action_length = self.controller_action_lengths[index]

            planner_hidden_idx = self.planner_hidden_idx[index]

            if self.split == 'train':
                if self.to_cache == True and index in self.img_data_cache:
                    img_feats = self.img_data_cache[index]
                else:
                    pos_queue = self.pos_queue[index]
                    images = self.get_frames(
                        self.env_loaded[self.env_list[index]],
                        pos_queue,
                        preprocess=True)
                    raw_img_feats = self.cnn(
                        Variable(torch.FloatTensor(images)
                                 .cuda())).data.cpu().numpy().copy()
                    img_feats = np.zeros(
                        (self.actions.shape[1], raw_img_feats.shape[1]),
                        dtype=np.float32)
                    img_feats[:raw_img_feats.shape[
                        0], :] = raw_img_feats.copy()
                    if self.to_cache == True:
                        self.img_data_cache[index] = img_feats

            if self.split in ['val', 'test'
                              ] or self.target_obj_conn_map_dir != False:
                target_obj_id, target_room = False, False
                bbox_obj = [
                    x for x in self.boxes[index]
                    if x['type'] == 'object' and x['target'] == True
                ][0]['box']
                for obj_id in self.env_loaded[self.env_list[index]].objects:
                    box2 = self.env_loaded[self.env_list[index]].objects[
                        obj_id]['bbox']
                    if all([bbox_obj['min'][x] == box2['min'][x] for x in range(3)]) == True and \
                        all([bbox_obj['max'][x] == box2['max'][x] for x in range(3)]) == True:
                        target_obj_id = obj_id
                        break
                bbox_room = [
                    x for x in self.boxes[index]
                    if x['type'] == 'room' and x['target'] == False
                ][0]
                for room in self.env_loaded[self.env_list[
                        index]].env.house.all_rooms:
                    if all([room['bbox']['min'][i] == bbox_room['box']['min'][i] for i in range(3)]) and \
                        all([room['bbox']['max'][i] == bbox_room['box']['max'][i] for i in range(3)]):
                        target_room = room
                        break
                assert target_obj_id != False
                assert target_room != False
                self.env_loaded[self.env_list[index]].set_target_object(
                    self.env_loaded[self.env_list[index]].objects[
                        target_obj_id], target_room)

                # [NOTE] only works for batch size = 1
                self.episode_pos_queue = self.pos_queue[index]
                self.episode_house = self.env_loaded[self.env_list[index]]
                self.target_room = target_room
                self.target_obj = self.env_loaded[self.env_list[
                    index]].objects[target_obj_id]

                return (idx, question, answer, actions, action_length)

            planner_pos_queue_idx = self.planner_pos_queue_idx[index]
            controller_pos_queue_idx = self.controller_pos_queue_idx[index]

            planner_img_feats = np.zeros(
                (self.actions.shape[1], img_feats.shape[1]), dtype=np.float32)
            planner_img_feats[:planner_action_length] = img_feats[
                planner_pos_queue_idx]

            planner_actions_in = planner_actions.clone() - 1
            planner_actions_out = planner_actions[1:].clone() - 2

            planner_actions_in[planner_action_length:].fill_(0)
            planner_mask = planner_actions_out.clone().gt(-1)
            if len(planner_actions_out) > planner_action_length:
                planner_actions_out[planner_action_length:].fill_(0)

            controller_img_feats = np.zeros(
                (self.actions.shape[1], img_feats.shape[1]), dtype=np.float32)
            controller_img_feats[:controller_action_length] = img_feats[
                controller_pos_queue_idx]

            controller_actions_in = actions[1:].clone() - 2
            if len(controller_actions_in) > controller_action_length:
                controller_actions_in[controller_action_length:].fill_(0)

            controller_out = controller_actions
            controller_mask = controller_out.clone().gt(-1)
            if len(controller_out) > controller_action_length:
                controller_out[controller_action_length:].fill_(0)

            return (idx, question, answer, planner_img_feats,
                    planner_actions_in, planner_actions_out,
                    planner_action_length, planner_mask, controller_img_feats,
                    controller_actions_in, planner_hidden_idx, controller_out,
                    controller_action_length, controller_mask)

    def __len__(self):
        if self.input_type == 'ques':
            return len(self.questions)
        else:
            return len(self.available_idx)


class EqaDataLoader(DataLoader):
    def __init__(self, **kwargs):
        if 'questions_h5' not in kwargs:
            raise ValueError('Must give questions_h5')
        if 'data_json' not in kwargs:
            raise ValueError('Must give data_json')
        if 'vocab' not in kwargs:
            raise ValueError('Must give vocab')
        if 'input_type' not in kwargs:
            raise ValueError('Must give input_type')
        if 'split' not in kwargs:
            raise ValueError('Must give split')
        if 'gpu_id' not in kwargs:
            raise ValueError('Must give gpu_id')

        questions_h5_path = kwargs.pop('questions_h5')
        data_json = kwargs.pop('data_json')
        input_type = kwargs.pop('input_type')

        split = kwargs.pop('split')
        vocab = kwargs.pop('vocab')

        gpu_id = kwargs.pop('gpu_id')

        if 'max_threads_per_gpu' in kwargs:
            max_threads_per_gpu = kwargs.pop('max_threads_per_gpu')
        else:
            max_threads_per_gpu = 10

        if 'to_cache' in kwargs:
            to_cache = kwargs.pop('to_cache')
        else:
            to_cache = False

        if 'target_obj_conn_map_dir' in kwargs:
            target_obj_conn_map_dir = kwargs.pop('target_obj_conn_map_dir')
        else:
            target_obj_conn_map_dir = False

        if 'map_resolution' in kwargs:
            map_resolution = kwargs.pop('map_resolution')
        else:
            map_resolution = 1000

        if 'image' in input_type or 'cnn' in input_type:
            kwargs['collate_fn'] = eqaCollateCnn
        elif 'lstm' in input_type:
            kwargs['collate_fn'] = eqaCollateSeq2seq

        print('Reading questions from ', questions_h5_path)
        with h5py.File(questions_h5_path, 'r') as questions_h5:
            self.dataset = EqaDataset(
                questions_h5,
                vocab,
                num_frames=kwargs.pop('num_frames'),
                data_json=data_json,
                split=split,
                gpu_id=gpu_id,
                input_type=input_type,
                max_threads_per_gpu=max_threads_per_gpu,
                to_cache=to_cache,
                target_obj_conn_map_dir=target_obj_conn_map_dir,
                map_resolution=map_resolution)

        super(EqaDataLoader, self).__init__(self.dataset, **kwargs)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_h5', default='data/04_22/train_v1.h5')
    parser.add_argument('-val_h5', default='data/04_22/val_v1.h5')
    parser.add_argument('-data_json', default='data/04_22/data_v1.json')
    parser.add_argument('-vocab_json', default='data/04_22/vocab_v1.json')

    parser.add_argument(
        '-input_type', default='ques', choices=['ques', 'ques,image'])
    parser.add_argument(
        '-num_frames', default=5,
        type=int)  # -1 = all frames of navigation sequence

    parser.add_argument('-batch_size', default=50, type=int)
    parser.add_argument('-max_threads_per_gpu', default=10, type=int)

    args = parser.parse_args()

    try:
        args.gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        args.gpus = [int(x) for x in args.gpus]
    except KeyError:
        print("CPU not supported")
        exit()

    train_loader_kwargs = {
        'questions_h5': args.train_h5,
        'data_json': args.data_json,
        'vocab': args.vocab_json,
        'batch_size': args.batch_size,
        'input_type': args.input_type,
        'num_frames': args.num_frames,
        'split': 'train',
        'max_threads_per_gpu': args.max_threads_per_gpu,
        'gpu_id': args.gpus[0],
        'cache_path': False,
    }

    train_loader = EqaDataLoader(**train_loader_kwargs)
    train_loader.dataset._load_envs(start_idx=0, in_order=True)
    t = 0

    while True:
        done = False
        all_envs_loaded = train_loader.dataset._check_if_all_envs_loaded()
        while done == False:
            print('[Size:%d][t:%d][Cache:%d]' %
                  (len(train_loader.dataset), t,
                   len(train_loader.dataset.img_data_cache)))
            for batch in train_loader:
                t += 1

            if all_envs_loaded == False:
                train_loader.dataset._load_envs(in_order=True)
                if len(train_loader.dataset.pruned_env_set) == 0:
                    done = True
            else:
                done = True