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
for test_ood in 'imagenet_v2' 'imagenet_a' 'imagenet_r' 'imagenet_s'
do
DIR=eval_closed_ood/CLIP/
CUDA_VISIBLE_DEVICES=0 python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--eval-only \
--output-dir ${DIR} \
test_ood ${test_ood}
done
done