#!/bin/bash

cd ../..
CUDA_LAUNCH_BLOCKING=1

# custom config
DATA=/path/to/datasets
TRAINER=BCAL

DATASET=$1
CFG=$2
CTP=end
NCTX=16
CSC=False
R=ViT_ep30

if [[$R = $CFG]]; then
backbone=VIT
else
backbone=RN50
fi

for TEST_ENV in "test_on_labelme.json" "test_on_sun.json"  "test_on_pascal.json"  "test_on_caltech.json"  
do
for SEED in 1 2 3
do
DIR=eval_open_ood/${TRAINER}/${DATASET}_${CFG}/seed${SEED}
python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/BCAL/${CFG}.yaml \
--eval-only \
--model-dir output/BCAL/VLCS_${backbone}/${TEST_ENV}/seed${SEED}/shots16 \
--output-dir ${DIR} \
TRAINER.BCAL.N_CTX ${NCTX} \
TRAINER.BCAL.CSC ${CSC} \
DATASET.NUM_SHOTS 16 \
TRAINER.BCAL.CLASS_TOKEN_POSITION ${CTP} \
TEST_ENV ${TEST_ENV} 
done
done

