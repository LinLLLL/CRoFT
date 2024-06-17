import os
import numpy as np
import random


os.system('CUDA_VISIBLE_DEVICES=6 bash test_imagenet.sh 0 32 1. 1.5')
os.system('CUDA_VISIBLE_DEVICES=6 bash test_imagenet.sh 0 32 1. 3.5')
os.system('CUDA_VISIBLE_DEVICES=6 bash test_imagenet.sh 0 32 1. 5.5')
os.system('CUDA_VISIBLE_DEVICES=6 bash test_imagenet.sh 0 32 1. 7.5')
os.system('CUDA_VISIBLE_DEVICES=6 bash test_imagenet.sh 0 32 1. 9.5')
os.system('CUDA_VISIBLE_DEVICES=6 bash test_imagenet.sh 0 32 1. 11.5')


os.system('CUDA_VISIBLE_DEVICES=6 bash test_imagenet.sh 0 32 0. 5.5')
os.system('CUDA_VISIBLE_DEVICES=6 bash test_imagenet.sh 0 32 0.5 5.5')
os.system('CUDA_VISIBLE_DEVICES=6 bash test_imagenet.sh 0 32 1. 5.5')
os.system('CUDA_VISIBLE_DEVICES=6 bash test_imagenet.sh 0 32 2. 5.5')
os.system('CUDA_VISIBLE_DEVICES=6 bash test_imagenet.sh 0 32 3. 5.5')
os.system('CUDA_VISIBLE_DEVICES=6 bash test_imagenet.sh 0 32 4. 5.5')
os.system('CUDA_VISIBLE_DEVICES=6 bash test_imagenet.sh 0 32 5. 5.5')




