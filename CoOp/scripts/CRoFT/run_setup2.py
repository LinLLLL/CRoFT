import os
import numpy as np
import random

for seed in [1,2,3]:
    for lambda2 in [1000, 100, 10, 1, 0]:
        os.system('CUDA_VISIBLE_DEVICES=0 bash croft_vlcs.sh VLCS rn50_ep30 ' + ' ' + str(seed) + ' ' + str(16) + ' ' + str(lambda2))
        os.system('CUDA_VISIBLE_DEVICES=0 bash croft_vlcs_vit.sh VLCS ViT_ep30 ' + ' ' + str(seed) + ' ' + str(16) + ' ' + str(lambda2))
        os.system('CUDA_VISIBLE_DEVICES=0 bash croft_pacs.sh PACS rn50_ep30 ' + ' ' + str(seed) + ' ' + str(16) + ' ' + str(lambda2))
        os.system('CUDA_VISIBLE_DEVICES=0 bash croft_pacs_vit.sh PACS ViT_ep30 ' + ' ' + str(seed) + ' ' + str(16) + ' ' + str(lambda2))
