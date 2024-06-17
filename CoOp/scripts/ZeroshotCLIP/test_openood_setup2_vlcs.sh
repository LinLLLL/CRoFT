#!/bin/bash

cd ../..
CUDA_LAUNCH_BLOCKING=1

# custom config
DATA=/path/to/datasets
TRAINER=ZeroshotCLIP

DATASET=$1
CFG=$2 


for TEST_ENV in  "test_on_pascal.json"  "test_on_labelme.json" "test_on_sun.json"   "test_on_caltech.json"  
do
for SEED in 1 2 3
do
DIR=eval_open_ood/${TRAINER}/${DATASET}_${CFG}/seed${SEED}
CUDA_VISIBLE_DEVICES=2 python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/CoOp/${CFG}.yaml \
--eval-only \
--output-dir ${DIR} \
TEST_ENV ${TEST_ENV} 
done
done