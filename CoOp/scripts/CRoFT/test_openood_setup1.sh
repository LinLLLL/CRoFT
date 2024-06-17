#!/bin/bash

cd ../..
CUDA_LAUNCH_BLOCKING=1

# custom config
DATA=/path/to/datasets
TRAINER=CRoFT

DATASET=imagenetood46
CFG=ViT_ep30
ratio=0.2
ratio_text=0.5

lambda1=$1
lambda2=$2
gpu_id=$3
shots=$4

for SEED in 1 2 3
do
DIR=eval_open_ood/CRoFT/shot${shots}
CUDA_VISIBLE_DEVICES=${gpu_id} python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--eval-only \
--model-dir output/CRoFT/imagenet46_${lambda1}_${lambda2}/seed${SEED}/shots${shots} \
--output-dir ${DIR} \
TRAINER.CRoFT.ratio_text ${ratio_text} \
TRAINER.CRoFT.ratio ${ratio} \
openood True
done

