import os
import numpy as np
import random

for seed in [1,2,3]:
    random.seed(seed)
    os.system('CUDA_VISIBLE_DEVICES=1 bash tip_pacs.sh PACS rn50_ep30 ' + ' ' + str(seed) + ' ' + str(16))
    os.system('CUDA_VISIBLE_DEVICES=1 bash tip_vlcs.sh VLCS rn50_ep30 ' + ' ' + str(seed) + ' ' + str(16))
    os.system('CUDA_VISIBLE_DEVICES=1 bash tip_pacs_vit.sh PACS ViT_ep30 ' + ' ' + str(seed) + ' ' + str(16))
    os.system('CUDA_VISIBLE_DEVICES=1 bash tip_vlcs_vit.sh VLCS ViT_ep30 ' + ' ' + str(seed) + ' ' + str(16))