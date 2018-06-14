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

def getListOfJsonFiles(json_dir):
    cmd = "find " + json_dir + " -name '*.json'"
    return subprocess.check_output(cmd, shell=True).strip().split()

def getTemplateNameFromPath(json_file_path):
    file_name = str(json_file_path).strip().split('/')[-1].split('.')[0]
    file_name_comps = file_name.strip().split('_')
    file_name_comps.pop(0)
    file_name_comps.pop(0)
    return "_".join(file_name_comps)

def getStats(json_dir):
    json_file_paths = getListOfJsonFiles(json_dir)
    master_ent = dict()

    for json_file_path in json_file_paths:
        template = getTemplateNameFromPath(json_file_path)
        master_ent[template] = dict()

        json_data = json.load(open(json_file_path, 'r'))
        for obj in json_data: master_ent[template][obj['ques']] = (obj['ent'], obj['count'])

    return master_ent

def getEnvWiseStats(qns_dataset, templates):
    env_wise_stats_json = {}
    house_ids = list(set([qn['house'] for qn in qns_dataset]))

    print ("Computing env-wise stats...")
    for i in tqdm(range(len(house_ids))):
        house_id = house_ids[i]
        qns_for_house = [qn for qn in qns_dataset if qn['house'] == house_id]

        # total unique questions (across all templates) before and after pruning
        before = len(set([qn['question'] for qn in qns_for_house]))
        after = len(set([qn['question'] for qn in qns_for_house if qn['accept']]))
        drop_rate = (before - after) / (1. * before)

        env_wise_stats_json[house_id] = {}
        env_wise_stats_json[house_id]['global'] = {
            'before': before,
            'after': after,
            'drop_rate': drop_rate
        }

        for template in templates:
            qns_for_template_for_house = [qn for qn in qns_for_house if collapseType(qn['type']) == template]
            before = len(set([qn['question'] for qn in qns_for_template_for_house]))
            after = len(set([qn['question'] for qn in qns_for_template_for_house if qn['accept']]))
            if before != 0.: drop_rate = (before - after) / (1. * before)
            else: drop_rate = 0.

            env_wise_stats_json[house_id][template] = {
                'before': before,
                'after': after,
                'drop_rate': drop_rate
            }

    return env_wise_stats_json

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

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
    		"-questionSet",
    		help="JSON file containing pre-generated list of questions to prune",
            default="data/question-engine-outputs/10_16/questions_original.json"
    )
    parser.add_argument(
        "-qnStatsJsonDir",
        help="Directory that has the JSON files containing question stats",
        default="data/entropy-stats/10_16/"
    )
    parser.add_argument(
        "-prunedOutputJson",
        help="Directory that the output JSONs (updated questions + env wise stats) are to be written to",
        default="data/question-engine-outputs/10_16/questions_pruned_countThresh=5.json"
    )
    parser.add_argument(
        "-envWiseStatsJson",
        help="Directory that the output JSONs (updated questions + env wise stats) are to be written to",
        default="data/env_wise_stats/10_16/env_wise_stats_countThresh=5.json"
    )
    args = parser.parse_args()

    master_ent = getStats(args.qnStatsJsonDir)
    qns_dataset = json.load(open(args.questionSet, "r"))
    templates = set(
        [ collapseType(qn['type']) for qn in qns_dataset ]
    )

    THRESH, COUNT = dict(), dict()
    for template in templates:
        THRESH[template] = 0.5
        COUNT[template] = 4

    for i in tqdm(range(len(qns_dataset))):
        qn = qns_dataset[i]
        assert collapseType(qn['type']) in master_ent
        assert collapseType(qn['type']) in THRESH

        if qn['question'] in master_ent[collapseType(qn['type'])]:
            ent = master_ent[collapseType(qn['type'])][qn['question']][0]
            count = master_ent[collapseType(qn['type'])][qn['question']][1]
            ent_thresh = THRESH[collapseType(qn['type'])]
            count_thresh = COUNT[collapseType(qn['type'])]

            if ent >= ent_thresh and count >= count_thresh: qn['accept'] = True
            else: qn['accept'] = False

            qn['entropy'] = ent
            qn['count'] = count

        # reject a question which has never been seen by the entropy engine
        # this will never happen as long as the entropy computation and
        # the filtering are being done on the same dataset of qns
        else:
            qn['accept'] = False
            qn['entropy'] = -1.0
            qn['count'] = 0

    # Save JSON qns_dataset (updated with 'accept' and 'entropy')
    with open(args.prunedOutputJson, "w") as f:
        json.dump(qns_dataset, f)

    # Stats
    # [global] Num of unique questions (across all templates, all envs) before and after pruning (also %-age drop)
    global_uniq_qns_before = len(set([qn['question'] for qn in qns_dataset]))
    global_uniq_qns_after = len(set([qn['question'] for qn in qns_dataset if qn['accept']]))
    global_reject = (global_uniq_qns_before - global_uniq_qns_after) / (1. * global_uniq_qns_before)

    # [template-wise] Num of unique questions per template (across all envs) before and after pruning (also %-age drop)
    for template in templates:
        uniq_qns_before = len(set([qn['question'] for qn in qns_dataset if collapseType(qn['type']) == template]))
        uniq_qns_after = len(set([qn['question'] for qn in qns_dataset if collapseType(qn['type']) == template and qn['accept'] == True]))

        if (uniq_qns_before == uniq_qns_after): reject_prop = 0.0
        else: reject_prop = (uniq_qns_before - uniq_qns_after) / (1. * uniq_qns_before)
        print ("%s, before = %d, after = %d, reject = %f" % (template, uniq_qns_before, uniq_qns_after, reject_prop))

    print ("\nTotal unique questions : before = %d, after = %d, reject = %f" % (global_uniq_qns_before, global_uniq_qns_after, global_reject))

    env_wise_stats = getEnvWiseStats(qns_dataset, templates)
    with open(args.envWiseStatsJson, "w") as f:
        json.dump(env_wise_stats, f)
