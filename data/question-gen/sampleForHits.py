# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import csv
import argparse
import operator
import os, sys, json, subprocess
from tqdm import tqdm
import numpy as np
import math

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def collapseType(q_type):
    collapse = {
        'exist_positive': 'exist',
        'exist_negative': 'exist',
        'exist_logical_positive':'exist_logic',
        'exist_logical_negative_1':'exist_logic',
        'exist_logical_negative_2':'exist_logic',
        'exist_logical_or_negative':'exist_logic',
        'exist_logical_or_positive_1': 'exist_logic',
        'exist_logical_or_positive_2': 'exist_logic',
        'dist_compare_positive': 'dist_compare',
        'dist_compare_negative': 'dist_compare'
    }

    if q_type in collapse: q_type = collapse[q_type]
    return q_type

"""
Given a template, the mapping should return the list of all
unique question strings that fall under the template.
Save this mapping for all templates in templates
"""
def getTemplateToQnStringMapping(question_set, templates):
    print ("Generating mapping from templates to (unique) question strings...")

    uniq_qns_by_template = {}
    for i in tqdm(range(len(templates))):
        template = templates[i]
        uniq_qns_by_template[template] = list(set([
            qn['question'] for qn in question_set
            if qn['accept'] and collapseType(qn['type']) == template
        ]))
    return uniq_qns_by_template

"""
Given a question string, the mapping should return the list of all instances
of that question string in the question set.
Save this mapping for all possible question strings in the question_set
"""
def getQnStringToInstancesMapping(question_set):

    print ("Generating mapping from qn str to qn instances")
    qn_str_to_instances = {}
    for i in tqdm(range(len(question_set))):
        qnObj = question_set[i]
        if not qnObj['accept']: continue

        qn_str = qnObj['question']
        if qn_str not in qn_str_to_instances: qn_str_to_instances[qn_str] = []
        qn_str_to_instances[qn_str].append(qnObj)

    return qn_str_to_instances

def getEnvWiseHitQnLimits(qns_per_env):
    qns_per_env_sorted = sorted(qns_per_env.items(), key=operator.itemgetter(1))
    num_envs = len(qns_per_env_sorted)

    return {
        'min': {
            'count': qns_per_env_sorted[0][1],
            'env': qns_per_env_sorted[0][0]
        },
        'max': {
            'count': qns_per_env_sorted[num_envs-1][1],
            'env': qns_per_env_sorted[num_envs-1][0]
        }
    }


def generateQuestionsForHITs(question_set):
    templates = list(set([collapseType(qn['type']) for qn in question_set]))
    uniq_qns_by_template = getTemplateToQnStringMapping(question_set, templates)
    qn_str_instances = getQnStringToInstancesMapping(question_set)

    qns_for_hits = []
    hit_qns_done_by_env = {}
    env_block_list = set()

    print ("Sampling questions for HITs...")
    # the loop should run for # unique (accepted) qns in question_set
    for i in range(10000):

        # check if there are any more questions to sample
        # if no, then break. if yes, uniformly sample a template
        if len(templates) == 0: break
        sampled_template = np.random.choice(templates)

        # check if there are questions left to be sampled from this template
        # if no, then remove the template from the templates list
        if len(uniq_qns_by_template[sampled_template]) == 0:
            templates.remove(sampled_template)
            continue

        # uniformly sample a question string from the sampled template
        # delete the question string so that it doesn't get sampled again
        sampled_qn_string = np.random.choice(uniq_qns_by_template[sampled_template])
        uniq_qns_by_template[sampled_template].remove(sampled_qn_string)

        # get instances of the sampled question string
        # add all question instances to the pool of questions for HITs
        # update the counts for the number of HIT questions per env
        # while adding all question instances, if the instance corresponds
        # to a env for which we already have 60 questions, do not add

        sampled_qn_string_instances = qn_str_instances[sampled_qn_string]
        for qnObj in sampled_qn_string_instances:
            if qnObj['house'] in env_block_list: continue

            if qnObj['house'] not in hit_qns_done_by_env:
                hit_qns_done_by_env[qnObj['house']] = 0
            hit_qns_done_by_env[qnObj['house']] += 1

            qns_for_hits.append(qnObj)

        # if any env has already reached 60 questions, add it to the block-list
        for env in hit_qns_done_by_env:
            if hit_qns_done_by_env[env] >= 60: env_block_list.add(env)

    # get the max and the min number of questions for any env
    print ("break at iteration %d" % i)
    limits = getEnvWiseHitQnLimits(hit_qns_done_by_env)
    print ("# envs = %d" % len(hit_qns_done_by_env))
    print ("%s has a min of %d qns" % (limits['min']['env'], limits['min']['count']))

    return qns_for_hits, hit_qns_done_by_env

def getJson(qns_for_hits, hits_per_env):
    envs_with_60_hits = [env for env in hits_per_env if hits_per_env[env] == 60]
    json_for_hits = {}
    for env in envs_with_60_hits:
        json_for_hits[env] = { 'templates': [], 'questions': [] }

    for qnObj in qns_for_hits:
        if qnObj['house'] not in envs_with_60_hits: continue
        json_for_hits[qnObj['house']]['questions'].append(qnObj)
        templates_set = set(json_for_hits[qnObj['house']]['templates'])
        templates_set.add(qnObj['type'])
        json_for_hits[qnObj['house']]['templates'] = list(templates_set)
    return json_for_hits


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-prunedQuestions",
        help="JSON file containing pruned set of questions",
        default="data/question-engine-outputs/10_16/questions_pruned_countThresh=5.json"
    )
    parser.add_argument(
        "-qnsForHitsJson",
        help="JSON file containing sampled questions for HITs",
        default="data/question-engine-outputs/10_16/questions_HITS_countThresh=5.json"
    )
    parser.add_argument(
        "-envWiseStatsJson",
        help="Directory that the output JSONs (updated questions + env wise stats) are to be written to",
        default="data/env_wise_stats/10_16/env_wise_stats_countThresh=5.json"
    )
    parser.add_argument(
        "-seed",
        help="PRNG seed",
        default=123456)
    args = parser.parse_args()

    question_set = json.load(open(args.prunedQuestions, "r"))
    print("%d questions loaded..." % len(question_set))
    questions_for_hits, hits_per_env = generateQuestionsForHITs(question_set)
    json_for_hits = getJson(questions_for_hits, hits_per_env)
    with open(args.qnsForHitsJson, "w") as f:
        json.dump(json_for_hits, f)
