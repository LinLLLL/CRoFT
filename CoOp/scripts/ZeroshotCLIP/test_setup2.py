import os
import numpy as np
import random


os.system('CUDA_VISIBLE_DEVICES=2 bash tsne_viz.sh PACS_DTD ViT_ep30')
os.system('CUDA_VISIBLE_DEVICES=2 bash tsne_viz.sh PACS_DTD rn50_ep30')
os.system('CUDA_VISIBLE_DEVICES=2 bash tsne_viz.sh PACS_food101 ViT_ep30')
os.system('CUDA_VISIBLE_DEVICES=2 bash tsne_viz.sh PACS_food101 rn50_ep30')
os.system('CUDA_VISIBLE_DEVICES=2 bash tsne_viz.sh PACS_Caltech101 ViT_ep30')
os.system('CUDA_VISIBLE_DEVICES=2 bash tsne_viz.sh PACS_Caltech101 rn50_ep30')


os.system('CUDA_VISIBLE_DEVICES=2 bash tsne_viz2.sh VLCS_Caltech101 ViT_ep30')
os.system('CUDA_VISIBLE_DEVICES=2 bash tsne_viz2.sh VLCS_Caltech101 rn50_ep30')
os.system('CUDA_VISIBLE_DEVICES=2 bash tsne_viz2.sh VLCS_food101 ViT_ep30')
os.system('CUDA_VISIBLE_DEVICES=2 bash tsne_viz2.sh VLCS_food101 rn50_ep30')
os.system('CUDA_VISIBLE_DEVICES=2 bash tsne_viz2.sh VLCS_DTD ViT_ep30')
os.system('CUDA_VISIBLE_DEVICES=2 bash tsne_viz2.sh VLCS_DTD rn50_ep30')
