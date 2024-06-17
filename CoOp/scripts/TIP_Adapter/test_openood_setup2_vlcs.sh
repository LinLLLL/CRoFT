#!/bin/bash

cd ../..
CUDA_LAUNCH_BLOCKING=1

# custom config
DATA=/path/to/datasets
TRAINER=TIP_ADAPTER


DATASET=$1
CFG=$2
train_data=$3


for TEST_ENV in "test_on_labelme.json" "test_on_sun.json"  "test_on_pascal.json"  "test_on_caltech.json"  
do
for SEED in 1 2 3
do
DIR=eval_open_ood/${TRAINER}/${DATASET}_${CFG}/seed${SEED}
python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--eval-only \
--model-dir output/TIP_ADAPTER/${train_data}_${CFG}/${TEST_ENV}/seed${SEED}/shots${TEST_ENV} \
--output-dir ${DIR} \
TEST_ENV ${TEST_ENV} \
TRAINER.TIP_ADAPTER.ALPHA 1. \
TRAINER.TIP_ADAPTER.BETA 10.
done
done



