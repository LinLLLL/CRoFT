#!/bin/bash

cd ../..
CUDA_LAUNCH_BLOCKING=1

# custom config
DATA=/path/to/datasets
TRAINER=ZeroshotCLIP

DATASET=imagenetood46
CFG=ViT_ep30


for SEED in 1 2 3
do
DIR=output/CLIP/seed${SEED}
CUDA_VISIBLE_DEVICES=0 python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/CoOp/${CFG}.yaml \
--eval-only \
--output-dir ${DIR} \
openood True
done