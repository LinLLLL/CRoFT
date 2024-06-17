import os
import numpy as np
import random

for seed in [1,2,3]:
    for lambda2 in [1000, 100, 10, 1, 0]:
        os.system('CUDA_VISIBLE_DEVICES=0 bash croft_pacs.sh PACS rn50_ep30 ' + ' ' + str(seed) + ' ' + str(16) + ' ' + str(lambda2))
        os.system('CUDA_VISIBLE_DEVICES=0 bash croft_pacs_vit.sh PACS ViT_ep30 ' + ' ' + str(seed) + ' ' + str(16) + ' ' + str(lambda2))


for seed in [1,2,3]:
    for lambda2 in [1000, 100, 10, 1, 0]:
        for lambda1 in [15, 5, 0]:
            for (ratio, ratio_text) in [(0.2, 0.5), (0.2, 0.0), (0.5, 0.0)]:
                os.system('CUDA_VISIBLE_DEVICES=0 bash croft_vlcs.sh VLCS rn50_ep30 ' + ' ' + str(seed) + ' ' + str(16) + ' ' + str(lambda2) + ' ' + str(lambda1) + ' ' + str(ratio)  + ' ' + str(ratio_text))
                os.system('CUDA_VISIBLE_DEVICES=0 bash croft_vlcs_vit.sh VLCS ViT_ep30 ' + ' ' + str(seed) + ' ' + str(16) + ' ' + str(lambda2) + ' ' + str(lambda1) + ' ' + str(ratio)  + ' ' + str(ratio_text))
