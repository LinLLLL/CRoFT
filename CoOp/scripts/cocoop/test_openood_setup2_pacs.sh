#!/bin/bash

cd ../..
CUDA_LAUNCH_BLOCKING=1

# custom config
DATA=/path/to/datasets
TRAINER=CoCoOp

DATASET=$1
CFG=$2  # ViT_ep30
train_data=$3



for TEST_ENV in "test_on_sketch.json"  "test_on_photo.json"  "test_on_cartoon.json"  "test_on_art_painting.json"  
do
for SEED in 1 2 3 
do
DIR=eval_open_ood/${TRAINER}/${DATASET}_${CFG}/seed${SEED}
python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER/${CFG}.yaml \
--model-dir output/COCOOP/${train_data}_${CFG}/${TEST_ENV}/seed${SEED}/16 \
--eval-only \
--output-dir ${DIR} \
TEST_ENV ${TEST_ENV} 
done
done