#!/bin/bash

cd ../..
CUDA_LAUNCH_BLOCKING=1

# custom config
DATA=/path/to/datasets
TRAINER=TIP_ADAPTER

DATASET=$1
CFG=$2  # config file
SEED=$3
SHOTS=$4


for TEST_ENV in "test_on_sketch.json"  "test_on_photo.json"  "test_on_cartoon.json"  "test_on_art_painting.json"  
do
DIR=output/TIP_ADAPTER/${DATASET}_${CFG}/${TEST_ENV}/seed${SEED}/shots${SHOTS}/
if [ -d "$DIR" ]; then
echo "Results are available in ${DIR}. Skip this job"
else
echo "Run this job and save the output to ${DIR}"
python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir ${DIR} \
DATASET.NUM_SHOTS ${SHOTS} \
TEST_ENV ${TEST_ENV} \
TRAINER.TIP_ADAPTER.ALPHA 1. \
TRAINER.TIP_ADAPTER.BETA 10.
fi
done