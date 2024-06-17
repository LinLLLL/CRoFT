#!/bin/bash

cd ../..
CUDA_LAUNCH_BLOCKING=1

# custom config
DATA=/path/to/datasets
TRAINER=BCAL

DATASET=$1
CFG=$2  # config file
SEED=$3
SHOTS=$4

CTP=end
NCTX=16
CSC=False

alpha1=0.01
alpha2=0.5
alpha3=0.01

for TEST_ENV in "test_on_sketch.json"  "test_on_photo.json"  "test_on_cartoon.json"  "test_on_art_painting.json"  
do
DIR=output/BCAL/${DATASET}_VIT/${TEST_ENV}/seed${SEED}/shots${SHOTS}/
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
TEST_ENV ${TEST_ENV} \
TRAINER.BCAL.N_CTX ${NCTX} \
TRAINER.BCAL.CSC ${CSC} \
TRAINER.BCAL.CLASS_TOKEN_POSITION ${CTP} \
DATASET.NUM_SHOTS ${SHOTS} \
alpha1 ${alpha1} \
alpha2 ${alpha2} \
alpha3 ${alpha3}
fi
done