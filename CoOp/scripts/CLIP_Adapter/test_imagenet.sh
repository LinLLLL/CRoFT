#!/bin/bash

cd ../..
CUDA_LAUNCH_BLOCKING=1

# custom config
DATA=/path/to/datasets
TRAINER=CLIP_Adapter

DATASET=imagenetood46
CFG=ViT_ep30
CTP=end
NCTX=16
CSC=False

gpu_id=$1
shots=$2

for SEED in 1 2 3
do
for test_ood in 'imagenet_v2' 'imagenet_a' 'imagenet_s'  'imagenet_r' 
do
DIR=eval_closed_ood/CLIP_Adapter/shot${shots}
CUDA_VISIBLE_DEVICES=${gpu_id} python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--eval-only \
--model-dir output/CLIP_Adapter/imagenet46/seed${SEED}/${shots} \
--output-dir ${DIR} \
test_ood ${test_ood} 
done
done

