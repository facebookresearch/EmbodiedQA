# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# adapted from https://github.com/facebookresearch/clevr-iep/blob/master/iep/preprocess.py

import h5py
import argparse
import numpy as np
from tqdm import tqdm
import os, sys, json, random

import pdb
"""
Tokenize a sequence, converting a string seq into a list of (string) tokens by
splitting on the specified delimiter. Optionally add start and end tokens.
"""


def tokenize(seq,
             delim=' ',
             punctToRemove=None,
             addStartToken=True,
             addEndToken=True):

    if punctToRemove is not None:
        for p in punctToRemove:
            seq = str(seq).replace(p, '')

    tokens = str(seq).split(delim)
    if addStartToken:
        tokens.insert(0, '<START>')

    if addEndToken:
        tokens.append('<END>')

    return tokens


def buildVocab(sequences,
               minTokenCount=1,
               delim=' ',
               punctToRemove=None,
               addSpecialTok=False):
    SPECIAL_TOKENS = {
        '<NULL>': 0,
        '<START>': 1,
        '<END>': 2,
        '<UNK>': 3,
    }

    tokenToCount = {}
    for seq in sequences:
        seqTokens = tokenize(
            seq,
            delim=delim,
            punctToRemove=punctToRemove,
            addStartToken=False,
            addEndToken=False)
        for token in seqTokens:
            if token not in tokenToCount:
                tokenToCount[token] = 0
            tokenToCount[token] += 1

    tokenToIdx = {}
    if addSpecialTok == True:
        for token, idx in SPECIAL_TOKENS.items():
            tokenToIdx[token] = idx
    for token, count in sorted(tokenToCount.items()):
        if count >= minTokenCount:
            tokenToIdx[token] = len(tokenToIdx)

    return tokenToIdx


def encode(seqTokens, tokenToIdx, allowUnk=False):
    seqIdx = []
    for token in seqTokens:
        if token not in tokenToIdx:
            if allowUnk:
                token = '<UNK>'
            else:
                raise KeyError('Token "%s" not in vocab' % token)
        seqIdx.append(tokenToIdx[token])
    return seqIdx


def decode(seqIdx, idxToToken, delim=None, stopAtEnd=True):
    tokens = []
    for idx in seqIdx:
        tokens.append(idxToToken[idx])
        if stopAtEnd and tokens[-1] == '<END>':
            break
    if delim is None:
        return tokens
    else:
        return delim.join(tokens)


def preprocessImages(obj, render_dir=False):
    working_dir = os.path.join(render_dir, 'working')
    path_id = obj['path_id']
    image_paths = []
    for i in range(len(obj['pos_queue']) - 1):
        image_paths.append('%s/%s_%05d.jpg' % (working_dir, path_id, i + 1))

    image_frames = []
    for i in image_paths:
        if os.path.isfile(i) == False:
            print(i)
            return False
        img = imread(i, mode='RGB')
        img = imresize(img, (224, 224), interp='bicubic')
        img = img.transpose(2, 0, 1)
        img = img / 255.0
        image_frames.append(img)
        # TODO: mean subtraction

    return image_frames


def processActions(actions):
    # from shortest-path-gen format
    # 0: forward
    # 1: left
    # 2: right
    # 3: stop
    #
    # to
    # 0: null
    # 1: start
    # 2: forward
    # 3: left
    # 4: right
    # 5: stop
    # for model training
    action_translations = {0: 2, 1: 3, 2: 4, 3: 5}

    action_ids = [1]

    for i in actions:
        action_ids.append(action_translations[i])
    return action_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_json', required=True)
    parser.add_argument('-input_vocab', default=None)
    parser.add_argument('-output_train_h5', required=True)
    parser.add_argument('-output_val_h5', required=True)
    parser.add_argument('-output_test_h5', required=True)
    parser.add_argument('-output_data_json', required=True)
    parser.add_argument('-output_vocab', default=None)
    parser.add_argument('-num_ques', default=10000000, type=int)
    parser.add_argument('-shortest_path_dir', required=True, type=str)
    args = parser.parse_args()

    random.seed(123)
    np.random.seed(123)

    assert args.input_vocab != None or args.output_vocab != None, "Either input or output vocab required"

    data = json.load(open(args.input_json, 'r'))

    houses = data['questions']
    questions = []

    for h in tqdm(houses):
        print(h, len(houses[h]))
        for q in houses[h]:
            if len(str(q['answer']).split(' ')) > 1:
                q['answer'] = '_'.join(q['answer'].split(' '))
            questions.append(q)

    print('Total questions: ', len(questions))

    # build vocab if no vocab file provided
    if args.input_vocab == None:
        answerTokenToIdx = buildVocab((str(q['answer']) for q in questions
                                       if q['answer'] != 'NIL'))
        questionTokenToIdx = buildVocab(
            (q['question'] for q in questions if q['answer'] != 'NIL'),
            punctToRemove=['?'],
            addSpecialTok=True)

        vocab = {
            'questionTokenToIdx': questionTokenToIdx,
            'answerTokenToIdx': answerTokenToIdx,
        }
    else:
        vocab = json.load(open(args.input_vocab, 'r'))

    if args.output_vocab != None:
        json.dump(vocab, open(args.output_vocab, 'w'))

    # encode questions
    idx, encoded_questions, question_types, answers, action_labels, action_lengths, pos_queue, envs, boxes = [], [], [], [], [], [], [], [], []
    for i, q in tqdm(enumerate(questions[:args.num_ques])):

        if os.path.exists(
                os.path.join(args.shortest_path_dir, q['house'] + '_' +
                             str(q['id']) + '.json')) == False:
            continue
        nav = json.load(
            open(
                os.path.join(args.shortest_path_dir, q['house'] + '_' +
                             str(q['id']) + '.json'), 'r'))

        idx.append(q['id'])
        questionTokens = tokenize(
            q['question'], punctToRemove=['?'], addStartToken=False)
        encoded_question = encode(questionTokens, vocab['questionTokenToIdx'])
        encoded_questions.append(encoded_question)
        question_types.append(q['type'])
        answers.append(vocab['answerTokenToIdx'][str(q['answer'])])

        # if there are 3 positions, there will be 2 actions + <stop>
        actions = nav['actions']
        positions = nav['positions']

        action_labels.append(processActions(actions))
        action_lengths.append(len(actions))

        pos_queue.append(positions)
        boxes.append(q['bbox'])

        envs.append(q['house'])

    args.num_ques = len(idx)
    maxALength = max(action_lengths) + 1

    action_labels_mat = np.zeros(
        (len(questions[:args.num_ques]), maxALength), dtype=np.int16)
    action_labels_mat.fill(0)  # 0 = null

    for i in tqdm(range(len(questions[:args.num_ques]))):
        for j in range(len(action_labels[i])):
            action_labels_mat[i][j] = action_labels[i][j]

    # pad encoded questions
    maxQLength = max(len(x) for x in encoded_questions)
    for qe in encoded_questions:
        while len(qe) < maxQLength:
            qe.append(vocab['questionTokenToIdx']['<NULL>'])

    # make train/test splits
    inds = list(range(0, len(idx)))
    random.shuffle(inds)

    train_envs = data['splits']['train']
    val_envs = data['splits']['val']
    test_envs = data['splits']['test']

    assert any([x in train_envs for x in test_envs]) == False
    assert any([x in train_envs for x in val_envs]) == False

    train_inds = [i for i in inds if envs[i] in train_envs]
    val_inds = [i for i in inds if envs[i] in val_envs]
    test_inds = [i for i in inds if envs[i] in test_envs]

    # TRAIN
    train_idx = [idx[i] for i in train_inds]
    train_encoded_questions = [encoded_questions[i] for i in train_inds]
    train_question_types = [question_types[i] for i in train_inds]
    train_answers = [answers[i] for i in train_inds]
    train_envs = [envs[i] for i in train_inds]
    train_pos_queue = [pos_queue[i] for i in train_inds]
    train_boxes = [boxes[i] for i in train_inds]

    train_action_labels = action_labels_mat[train_inds]
    train_action_lengths = [action_lengths[i] for i in train_inds]

    # VAL
    val_idx = [idx[i] for i in val_inds]
    val_encoded_questions = [encoded_questions[i] for i in val_inds]
    val_question_types = [question_types[i] for i in val_inds]
    val_answers = [answers[i] for i in val_inds]
    val_envs = [envs[i] for i in val_inds]
    val_pos_queue = [pos_queue[i] for i in val_inds]
    val_boxes = [boxes[i] for i in val_inds]

    val_action_labels = action_labels_mat[val_inds]
    val_action_lengths = [action_lengths[i] for i in val_inds]

    # TEST
    test_idx = [idx[i] for i in test_inds]
    test_encoded_questions = [encoded_questions[i] for i in test_inds]
    test_question_types = [question_types[i] for i in test_inds]
    test_answers = [answers[i] for i in test_inds]
    test_envs = [envs[i] for i in test_inds]
    test_pos_queue = [pos_queue[i] for i in test_inds]
    test_boxes = [boxes[i] for i in test_inds]

    test_action_labels = action_labels_mat[test_inds]
    test_action_lengths = [action_lengths[i] for i in test_inds]

    # parse envs
    all_envs = list(set(envs))
    train_env_idx = [all_envs.index(x) for x in train_envs]
    val_env_idx = [all_envs.index(x) for x in val_envs]
    test_env_idx = [all_envs.index(x) for x in test_envs]

    # write h5 files
    print('Writing hdf5')

    train_encoded_questions = np.asarray(
        train_encoded_questions, dtype=np.int16)
    print('Train', train_encoded_questions.shape)
    with h5py.File(args.output_train_h5, 'w') as f:
        f.create_dataset('idx', data=np.asarray(train_idx))
        f.create_dataset('questions', data=train_encoded_questions)
        f.create_dataset('answers', data=np.asarray(train_answers))
        f.create_dataset(
            'action_labels',
            data=np.asarray(train_action_labels),
            dtype=np.int16)
        f.create_dataset(
            'action_lengths',
            data=np.asarray(train_action_lengths),
            dtype=np.int16)

    val_encoded_questions = np.asarray(val_encoded_questions, dtype=np.int16)
    print('Val', val_encoded_questions.shape)
    with h5py.File(args.output_val_h5, 'w') as f:
        f.create_dataset('idx', data=np.asarray(val_idx))
        f.create_dataset('questions', data=val_encoded_questions)
        f.create_dataset('answers', data=np.asarray(val_answers))
        f.create_dataset(
            'action_labels',
            data=np.asarray(val_action_labels),
            dtype=np.int16)
        f.create_dataset(
            'action_lengths',
            data=np.asarray(val_action_lengths),
            dtype=np.int16)

    test_encoded_questions = np.asarray(test_encoded_questions, dtype=np.int16)
    print('Test', test_encoded_questions.shape)
    with h5py.File(args.output_test_h5, 'w') as f:
        f.create_dataset('idx', data=np.asarray(test_idx))
        f.create_dataset('questions', data=test_encoded_questions)
        f.create_dataset('answers', data=np.asarray(test_answers))
        f.create_dataset(
            'action_labels',
            data=np.asarray(test_action_labels),
            dtype=np.int16)
        f.create_dataset(
            'action_lengths',
            data=np.asarray(test_action_lengths),
            dtype=np.int16)

    json.dump({
        'envs': all_envs,
        'train_env_idx': train_env_idx,
        'val_env_idx': val_env_idx,
        'test_env_idx': test_env_idx,
        'train_pos_queue': train_pos_queue,
        'val_pos_queue': val_pos_queue,
        'test_pos_queue': test_pos_queue,
        'train_boxes': train_boxes,
        'val_boxes': val_boxes,
        'test_boxes': test_boxes
    }, open(args.output_data_json, 'w'))
