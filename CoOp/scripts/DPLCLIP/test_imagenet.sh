#!/bin/bash

cd ../..
CUDA_LAUNCH_BLOCKING=1

# custom config
DATA=/path/to/datasets
TRAINER=DPLCLIP

DATASET=imagenetood46
CFG=ViT_ep30

CTP=end
NCTX=16
CSC=False

gpu_id=$1
shots=$2

for SEED in 1 2 3
do
for test_ood in 'imagenet_v2' 'imagenet_a' 'imagenet_r' 'imagenet_s'
do
DIR=eval_closed_ood/DPLCLIP/shot${shots}
CUDA_VISIBLE_DEVICES=${gpu_id} python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--eval-only \
--model-dir output/DPLCLIP/imagenet46/seed${SEED}/shots${shots} \
--output-dir ${DIR} \
TRAINER.DPLCLIP.N_CTX ${NCTX} \
TRAINER.DPLCLIP.CSC ${CSC} \
TRAINER.DPLCLIP.CLASS_TOKEN_POSITION ${CTP} \
test_ood ${test_ood}
done
done