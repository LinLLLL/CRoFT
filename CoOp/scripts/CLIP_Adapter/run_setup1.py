import os
import numpy as np
import random

CSC = 'False'
CTP = 'end'

for seed in [2,3,1]:
    for shots in [4]:
        os.system('CUDA_VISIBLE_DEVICES=0 bash adapter.sh imagenet46 ViT_ep30 ' + str(seed) + ' ' + 'CLIP_Adapter' + ' ' + str(shots))


