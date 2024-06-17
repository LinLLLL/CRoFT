import os
import numpy as np
import random

for seed in [1,2,3]:
    random.seed(seed)
    for lambda1 in [1, 5, 10, 15, 20]:
        for lambda2 in [1, 5, 10, 15, 20]:
            os.system('CUDA_VISIBLE_DEVICES=0 bash croft.sh imagenet46 ViT_ep30 ' + ' ' + str(seed) + ' ' + str(1) + ' ' + str(lambda1) + ' ' + str(lambda2) + ' ' + str(10))
            os.system('CUDA_VISIBLE_DEVICES=0 bash croft.sh imagenet46 ViT_ep30 ' + ' ' + str(seed) + ' ' + str(16) + ' ' + str(lambda1) + ' ' + str(lambda2) + ' ' + str(1))
            os.system('CUDA_VISIBLE_DEVICES=0 bash croft.sh imagenet46 ViT_ep30 ' + ' ' + str(seed) + ' ' + str(32) + ' ' + str(lambda1) + ' ' + str(lambda2) + ' ' + str(1))


