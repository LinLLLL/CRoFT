import os
import numpy as np
import random


for seed in [1, 2, 3]:
    for shots in [1, 16, 32]:
        os.system('CUDA_VISIBLE_DEVICES=0 bash bcal.sh imagenet46 ViT_ep30 ' + str(seed) + ' ' + 'BCAL' + ' ' + str(shots) + ' ' + str(0.01) + ' ' + str(0.5) + ' ' + str(0.01))    
