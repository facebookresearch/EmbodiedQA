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

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "-hitsJson",
            help="JSON file containing sampled questions for HITs",
            default="data/question-engine-outputs/10_16/questions_HITS_countThresh=4.json"
    )
    parser.add_argument(
            "-outputFile",
            help="Text file containing questions sampled for HITs",
            default="data/question_samples/10_20/sampleQns_countThresh=4_entThresh=0.5.txt"
    )
    parser.add_argument(
        "-seed",
        help="PRNG seed",
        default=123456)
    args = parser.parse_args()

    prep_templates = [
    	'above', 'on', 'below', 'under', 'next_to', 'above_room',
    	'on_room', 'below_room', 'under_room', 'next_to_room'
    ]

    sampled_questions = json.load(open(args.hitsJson, "r"))
    with open(args.outputFile, "w") as f:
    	for house_id in sampled_questions:
    		f.write("\n")
    		f.write(house_id)
    		f.write("\n")

    		qns_from_this_house = sampled_questions[house_id]['questions']
    		for qn in qns_from_this_house:
    			line = "\t(" + qn['question'] + ", " + str(qn['answer']) + ")"
    			f.write(line)
    			f.write("\n")
