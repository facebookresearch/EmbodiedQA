#!/bin/sh

if [ $# -eq 0 ]
	then
		echo "Usage: ./run_me.sh mm_dd\n"
		exit 1
fi

ROOT=`pwd`
SUNCG_DATA_DIR=/path/to/suncg
ENV_SET_JSON=data/envs.json

########################################
# Directories
########################################
QNS_JSON_DIR=data/question-engine-outputs
ENTROPY_STATS_DIR=data/entropy-stats
ENV_WISE_STATS_DIR=data/env_wise_stats
QNS_SAMPLES_DIR=data/question_samples

if [ ! -d "${QNS_JSON_DIR}/$1" ]; then
	mkdir -p "${QNS_JSON_DIR}/$1"
fi

if [ ! -d "${ENTROPY_STATS_DIR}/$1" ]; then
	mkdir -p "${ENTROPY_STATS_DIR}/$1"
fi

if [ ! -d "${ENV_WISE_STATS_DIR}/$1" ]; then
	mkdir -p "${ENV_WISE_STATS_DIR}/$1"
fi

if [ ! -d "${QNS_SAMPLES_DIR}/$1" ]; then
	mkdir -p "${QNS_SAMPLES_DIR}/$1"
fi

########################################
# Files
########################################
ORIG_QUESTION_SET=${QNS_JSON_DIR}/$1/questions_original.json
ENTROPY_STATS_DIR=${ENTROPY_STATS_DIR}/$1/
PRUNED_QUESTION_SET=${QNS_JSON_DIR}/$1/questions_pruned_entThresh=0.5_countThresh=4.json
ENV_WISE_STATS=${ENV_WISE_STATS_DIR}/$1/env_wise_stats_countThresh=4.json
# HITS_QUESTION_SET=${QNS_JSON_DIR}/$1/questions_HITS_entThresh=0.5_countThresh=4.json
# MANUAL_VIEW_TXT=${QNS_SAMPLES_DIR}/$1/sampleQns_countThresh=4_entThresh=0.5.txt

echo "\n----------------------------------------"
echo "Generating questions for environments..."
echo "----------------------------------------\n"
python engine.py \
	-dataDir ${SUNCG_DATA_DIR} \
	-inputJson ${ENV_SET_JSON} \
	-outputJson ${ORIG_QUESTION_SET}

echo "\n----------------------------------------"
echo "Computing entropy stats for all questions..."
echo "----------------------------------------\n"
python entro.py \
	-inputJson ${ORIG_QUESTION_SET} \
	-outputDir ${ENTROPY_STATS_DIR}

echo "\n----------------------------------------"
echo "Filtering based on entropy+count thresholds"
echo "----------------------------------------\n"
python entropy_based_filtering.py \
	-questionSet ${ORIG_QUESTION_SET} \
	-qnStatsJsonDir ${ENTROPY_STATS_DIR} \
	-prunedOutputJson ${PRUNED_QUESTION_SET} \
	-envWiseStatsJson ${ENV_WISE_STATS}

# echo "\n----------------------------------------"
# echo "Sample questions for HITs..."
# echo "----------------------------------------\n"
# python sampleForHits.py \
# 	-prunedQuestions ${PRUNED_QUESTION_SET} \
# 	-qnsForHitsJson ${HITS_QUESTION_SET} \
# 	-envWiseStatsJson ${ENV_WISE_STATS} \

# echo "\n----------------------------------------"
# echo "Generating text file for manual viewing..."
# echo "----------------------------------------\n"
# python sampleForManualView.py \
# 	-hitsJson ${HITS_QUESTION_SET} \
# 	-outputFile ${MANUAL_VIEW_TXT}
