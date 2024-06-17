cd ../..
CUDA_LAUNCH_BLOCKING=1


DATA=/path/to/datasets
TRAINER=CRoFT

DATASET=imagenetood46
CFG=ViT_ep30
ratio=0.2
ratio_text=0.5


for SEED in 1 2 3
do
for test_ood in 'imagenet_v2' 'imagenet_a' 'imagenet_s'  'imagenet_r' 
do
DIR=output/CRoFT/energy/shot32
CUDA_VISIBLE_DEVICES=0 python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--eval-only \
--model-dir output/CRoFT/imagenet46/seed${SEED}/shots32 \
--output-dir ${DIR} \
TRAINER.CRoFT.ratio_text ${ratio_text} \
TRAINER.CRoFT.ratio ${ratio} \
test_ood ${test_ood} \
load_energy True
done
done