import os
import numpy as np
import random


os.system('CUDA_VISIBLE_DEVICES=1 bash test_openood_setup2_vlcs.sh VLCS_DTD ViT_ep30 VLCS')
os.system('CUDA_VISIBLE_DEVICES=1 bash test_openood_setup2_vlcs.sh VLCS_DTD rn50_ep30 VLCS')
os.system('CUDA_VISIBLE_DEVICES=1 bash test_openood_setup2_vlcs.sh VLCS_food101 ViT_ep30 VLCS')
os.system('CUDA_VISIBLE_DEVICES=1 bash test_openood_setup2_vlcs.sh VLCS_food101 rn50_ep30 VLCS')
os.system('CUDA_VISIBLE_DEVICES=1 bash test_openood_setup2_vlcs.sh VLCS_Caltech101 ViT_ep30 VLCS')
os.system('CUDA_VISIBLE_DEVICES=1 bash test_openood_setup2_vlcs.sh VLCS_Caltech101 rn50_ep30 VLCS')


os.system('CUDA_VISIBLE_DEVICES=1 bash test_openood_setup2_pacs.sh PACS_DTD ViT_ep30 PACS')
os.system('CUDA_VISIBLE_DEVICES=1 bash test_openood_setup2_pacs.sh PACS_DTD rn50_ep30 PACS')
os.system('CUDA_VISIBLE_DEVICES=1 bash test_openood_setup2_pacs.sh PACS_food101 ViT_ep30 PACS')
os.system('CUDA_VISIBLE_DEVICES=1 bash test_openood_setup2_pacs.sh PACS_food101 rn50_ep30 PACS')
os.system('CUDA_VISIBLE_DEVICES=1 bash test_openood_setup2_pacs.sh PACS_Caltech101 ViT_ep30 PACS')
os.system('CUDA_VISIBLE_DEVICES=1 bash test_openood_setup2_pacs.sh PACS_Caltech101 rn50_ep30 PACS')
