#!/bin/bash

cd ../..
CUDA_LAUNCH_BLOCKING=1

# custom config
DATA=/path/to/datasets
TRAINER=CRoFT

DATASET=$1
CFG=$2  # config file
SEED=$3
SHOTS=$4
lambda1=$5
lambda2=$6
ratio=0.2
ratio_text=0.5
gen_step=$7


DIR=output/CRoFT/${DATASET}_${lambda1}_${lambda2}/seed${SEED}/shots${SHOTS}/
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
TRAINER.CRoFT.WCCF True \
TRAINER.CRoFT.gen_step ${gen_step} \
TRAINER.CRoFT.ratio ${ratio} \
TRAINER.CRoFT.ratio_text ${ratio_text} \
TRAINER.CRoFT.lambda1 ${lambda1} \
TRAINER.CRoFT.lambda2 ${lambda2}
fi