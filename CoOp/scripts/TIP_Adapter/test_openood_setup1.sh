#!/bin/bash

cd ../..
CUDA_LAUNCH_BLOCKING=1

# custom config
DATA=/path/to/datasets
TRAINER=TIP_ADAPTER


DATASET=imagenetood46
CFG=ViT_ep30

gpu_id=$1
shots=$2
alpha=$3
beta=$4


for SEED in 1 2 3
do
DIR=eval_open_ood/TIP_ADAPTER/shot${shots}
CUDA_VISIBLE_DEVICES=${gpu_id} python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--eval-only \
--model-dir output/TIP_ADAPTER/imagenet46/seed${SEED}/shots${shots} \
--output-dir ${DIR} \
openood True \
DATASET.NUM_SHOTS ${shots} \
TRAINER.TIP_ADAPTER.ALPHA ${alpha} \
TRAINER.TIP_ADAPTER.BETA ${beta}
done