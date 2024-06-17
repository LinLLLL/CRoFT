import os
import numpy as np
import random

for seed in [1, 2, 3]:
    random.seed(seed)
    os.system('CUDA_VISIBLE_DEVICES=1 bash tip.sh imagenet46 ViT_ep30 ' + ' ' + str(seed) + ' ' + str(1))
    os.system('CUDA_VISIBLE_DEVICES=1 bash tip.sh imagenet46 ViT_ep30 ' + ' ' + str(seed) + ' ' + str(16))
    os.system('CUDA_VISIBLE_DEVICES=1 bash tip.sh imagenet46 ViT_ep30 ' + ' ' + str(seed) + ' ' + str(32))
