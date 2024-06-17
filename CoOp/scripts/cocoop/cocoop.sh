#!/bin/bash

cd ../..
CUDA_LAUNCH_BLOCKING=1

# custom config
DATA=/path/to/datasets

DATASET=$1
CFG=$2  # config file
SEED=$3
TRAINER=$4
SHOTS=$5

CTP=end
NCTX=16
CSC=False


DIR=output/COCOOP/${DATASET}/seed${SEED}/shots${SHOTS}/
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
TRAINER.COCOOP.N_CTX ${NCTX} \
TRAINER.COCOOP.CSC ${CSC} \
TRAINER.COCOOP.CLASS_TOKEN_POSITION ${CTP} \
DATASET.NUM_SHOTS ${SHOTS} 
fi