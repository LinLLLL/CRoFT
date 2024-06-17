import pickle
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('AGG')
import sys
import os
print(os.getcwd())
sys.path.append(os.getcwd())
sys.path.append("../..")

import matplotlib.pyplot as plt
from metric_utils import *
import sklearn
from sklearn import covariance
from metric_utils import get_measures


recall_level_default = 0.95
length = 512

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()


# ID data

normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
prepos_feat = lambda x: np.ascontiguousarray(np.concatenate([normalizer(x)], axis=1))

dict = {}
for seed in [1,2,3]:
    seed = str(seed)
    for openood_name in ['DTD', 'food101', 'Caltech101']:

        dict[openood_name] = {}
        dict[openood_name]['auroc'] = {}
        dict[openood_name]['aupr'] = {}
        dict[openood_name]['fpr'] = {}

        # for test_env in ["caltech", "pascal", "labelme", "sun"]:
        for test_env in ["sketch", "photo", "cartoon",  "art_painting"]:

            id_train_data = np.load('../../output/CRoFT_ImgFeat/CRoFT/SEED{}/test_on_{}_ViT_PACS_{}_ID.npy'.format(seed, test_env, openood_name))
            print(id_train_data.shape)
            id_train_data = torch.from_numpy(id_train_data).reshape(-1, length)
            print(id_train_data.shape)

            all_data_in = np.load('../../output/CRoFT_ImgFeat/CRoFT/SEED{}/test_on_{}_ViT_PACS_{}_COOD.npy'.format(seed, test_env, openood_name))
            all_data_in = torch.from_numpy(all_data_in).reshape(-1, length)

            all_data_out = np.load('../../output/CRoFT_ImgFeat/CRoFT/SEED{}/test_on_{}_ViT_PACS_{}_SOOD.npy'.format(seed, test_env, openood_name))
            all_data_out = torch.from_numpy(all_data_out).reshape(-1, length)


            all_data_in = all_data_in.reshape(-1, length).numpy()
            all_data_out = all_data_out.reshape(-1, length).numpy()
            print(all_data_in.shape)
            print(all_data_out.shape)

            id = 0
            T = 1
            scores_in = []
            scores_ood_test = []

            mean_list = []
            covariance_list = []


            # knn score
            id_train_data = prepos_feat(id_train_data)
            all_data_in = prepos_feat(all_data_in)
            all_data_out = prepos_feat(all_data_out)


            import faiss
            index = faiss.IndexFlatL2(id_train_data.shape[1])
            index.add(id_train_data)
            # index.add(id_train_data)
            for K in [1, 5, 10, 25, 50, 100, 200, 300, 400, 500]:
                D, _ = index.search(all_data_in, K)
                print(D.shape)
                scores_in = -D[:,-1]
                all_results = []
                all_score_ood = []
                # for ood_dataset, food in food_all.items():
                D, _ = index.search(all_data_out, K)
                scores_ood_test = -D[:,-1]
                all_score_ood.extend(scores_ood_test)


            #    print('-----------------id------------------')
            #    print(np.percentile(scores_in, 5))
            #    print(np.percentile(scores_in, 25))
            #    print(np.percentile(scores_in, 50))
            #    print(np.percentile(scores_in, 75))
            #    print(np.percentile(scores_in, 95))
            #    print('-----------------ood------------------')
            #    print(np.percentile(scores_ood_test, 5))
            #    print(np.percentile(scores_ood_test, 25))
            #    print(np.percentile(scores_ood_test, 50))
            #    print(np.percentile(scores_ood_test, 75))
            #    print(np.percentile(scores_ood_test, 95))

                results = get_measures(scores_in, scores_ood_test, plot=False)
                print_measures(results[0], results[1], results[2], f'KNN k={K}')
                if str(K) not in dict[openood_name]['auroc'].keys():
                    dict[openood_name]['auroc'][str(K)] = []
                    dict[openood_name]['aupr'][str(K)] = []
                    dict[openood_name]['fpr'][str(K)] = []
                dict[openood_name]['auroc'][str(K)].append(results[0])
                dict[openood_name]['aupr'][str(K)].append(results[1])
                dict[openood_name]['fpr'][str(K)].append(results[2])

for openood_name in dict.keys():
    for K in dict[openood_name]['auroc'].keys():
        print('------------------------------------- K={} ------------------------------------'.format(K))
        print('auroc on {} is {}'.format(openood_name, np.mean(dict[openood_name]['auroc'][K])))
        print('aupr on {} is {}'.format(openood_name, np.mean(dict[openood_name]['aupr'][K])))
        print('fpr on {} is {}'.format(openood_name, np.mean(dict[openood_name]['fpr'][K])))
        print('-------------------------------------------------------------------------------')
        print()


