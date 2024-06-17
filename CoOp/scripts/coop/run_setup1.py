import os
import numpy as np
import random

CSC = 'False'
CTP = 'end'

 for seed in [1, 2, 3]:
     for shots in [1, 16, 32]:
         os.system('CUDA_VISIBLE_DEVICES=0 bash coop.sh imagenet46 ViT_ep30 ' + str(seed) + ' ' + 'CoOp' + ' ' + str(shots))
