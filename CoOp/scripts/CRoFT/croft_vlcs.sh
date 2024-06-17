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
lambda2=$5
lambda1=$6
ratio=$7
ratio_text=$8


for TEST_ENV in "test_on_labelme.json" "test_on_sun.json"  "test_on_pascal.json" "test_on_caltech.json"  
do
DIR=output/CRoFT/${DATASET}_${CFG}_${lambda1}_${lambda2}/${TEST_ENV}/seed${SEED}/shots${SHOTS}/
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
TRAINER.CRoFT.gen_step 5 \
TEST_ENV ${TEST_ENV} \
TRAINER.CRoFT.lambda1 ${lambda1} \
TRAINER.CRoFT.lambda2 ${lambda2} \
TRAINER.CRoFT.ratio ${ratio} \
TRAINER.CRoFT.ratio_text ${ratio_text} 
fi
done