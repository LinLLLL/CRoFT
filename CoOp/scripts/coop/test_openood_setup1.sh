#!/bin/bash

cd ../..
CUDA_LAUNCH_BLOCKING=1

# custom config
DATA=/path/to/datasets
TRAINER=CoOp

DATASET=imagenetood46
CFG=ViT_ep30

gpu_id=$1
shots=$2


DIR=eval_open_ood/COOP/shot${shots}
for SEED in 1 2 3
do
CUDA_VISIBLE_DEVICES=${gpu_id} python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--eval-only \
--model-dir output/COOP/imagenet46/seed${SEED}/shots${shots} \
--output-dir ${DIR} \
openood True
done