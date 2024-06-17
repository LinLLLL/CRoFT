import os
import numpy as np
import random

CSC = 'False'
CTP = 'end'

for seed in [1, 2, 3]:
    for shots in [16]:
        os.system('CUDA_VISIBLE_DEVICES=0 bash coop_vlcs.sh VLCS ViT_ep30 ' + str(seed) + ' ' + 'CoOp' + ' ' + str(shots))
        os.system('CUDA_VISIBLE_DEVICES=0 bash coop_pacs.sh PACS ViT_ep30 ' + str(seed) + ' ' + 'CoOp' + ' ' + str(shots))
        os.system('CUDA_VISIBLE_DEVICES=0 bash coop_vlcs.sh VLCS rn50_ep30 ' + str(seed) + ' ' + 'CoOp' + ' ' + str(shots))
        os.system('CUDA_VISIBLE_DEVICES=0 bash coop_pacs.sh PACS rn50_ep30 ' + str(seed) + ' ' + 'CoOp' + ' ' + str(shots))