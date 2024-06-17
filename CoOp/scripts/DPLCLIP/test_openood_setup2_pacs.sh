#!/bin/bash

cd ../..
CUDA_LAUNCH_BLOCKING=1

# custom config
DATA=/path/to/datasets
TRAINER=DPLCLIP


DATASET=$1
CFG=$2
CTP=end
NCTX=16
CSC=False


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
--config-file configs/trainers/FRYK/${CFG}.yaml \
--eval-only \
--model-dir output/DPLCLIP/PACS_${CFG}/${TEST_ENV}/seed${SEED}/16 \
--output-dir ${DIR} \
TRAINER.DPLCLIP.N_CTX ${NCTX} \
TRAINER.DPLCLIP.CSC ${CSC} \
DATASET.NUM_SHOTS 16 \
TRAINER.DPLCLIP.CLASS_TOKEN_POSITION ${CTP} \
TEST_ENV ${TEST_ENV} 
done
done



