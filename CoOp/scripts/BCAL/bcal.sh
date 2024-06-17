#!/bin/bash
cd ../..
#CUDA_VISIBLE_DEVICES=2
CUDA_LAUNCH_BLOCKING=1

# custom config
DATA=/path/to/datasets

DATASET=$1
CFG=$2  # config file
SEED=$3
TRAINER=$4
SHOTS=$5
alpha1=$6
alpha2=$7
alpha3=$8

CTP=end
NCTX=16
CSC=False

DIR=output/${TRAINER}/${DATASET}/seed${SEED}/${SHOTS}/

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
TRAINER.BCAL.N_CTX ${NCTX} \
TRAINER.BCAL.CSC ${CSC} \
TRAINER.BCAL.CLASS_TOKEN_POSITION ${CTP} \
DATASET.NUM_SHOTS ${SHOTS} \
alpha1 ${alpha1} \
alpha2 ${alpha2} \
alpha3 ${alpha3}
fi