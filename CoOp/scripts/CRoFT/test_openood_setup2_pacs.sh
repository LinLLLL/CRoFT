#!/bin/bash

cd ../..
CUDA_LAUNCH_BLOCKING=1

# custom config
DATA=/path/to/datasets
TRAINER=CRoFT

DATASET=$1
CFG=$2
ratio=0.5
ratio_text=0.0


for TEST_ENV in "test_on_sketch.json"  "test_on_photo.json"  "test_on_cartoon.json"  "test_on_art_painting.json"  
do
for SEED in 1 2 3
do
if [ "$CFG" == "ViT_ep30" ] ;then
model=PACS_ViT_ep30_5_1000  # the selected best model name
else
model=PACS_rn50_ep30_5_1000  # the selected best model name
fi
DIR=eval_open_ood/${TRAINER}/${DATASET}_${CFG}/seed${SEED}
python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--eval-only \
--model-dir output/CRoFT/${model}/${TEST_ENV}/seed${SEED}/shots16 \
--output-dir ${DIR} \
TRAINER.CRoFT.ratio_text ${ratio_text} \
TRAINER.CRoFT.ratio ${ratio} \
TEST_ENV ${TEST_ENV} 
done
done