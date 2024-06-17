import os
import numpy as np
import pandas as pd

path = 'eval_open_ood/CRoFT'
files = os.listdir(path)

dict = {}
dict1 = {}
for seed in range(1):
    for file in files:
        if "shot" not in file:
            continue
        log_path = os.path.join(path, file)
        shot_files = os.listdir(log_path)
        res_fpr = []
        res_auroc = []
        for log in shot_files:
            file_path = os.path.join(log_path, log)
            print(file_path)
            with open(file_path, 'r') as f:
                res_fpr_tmp = []
                for line in f.readlines():
                    if 'Loading weights to generator from' in line:
                        seed = line.split("seed")[-1].split("/")[0].strip()
                        l1 = line.split("imagenet46")[-1].split("_")[1].strip()
                        l2 = line.split("imagenet46")[-1].split("_")[2].split("/seed")[0].strip()
                    if 'OODD_FPR95:' in line:
                        res_fpr.append(np.float32(line.split(': ')[-1].strip()))
                    if 'AUROC:' in line:
                        res_auroc.append(np.float32(line.split(':')[-1].strip()))
            print('-------------------------------------------------------------------')
            print(file + "_" + l1 + "_" + l2, np.mean(res_fpr))
            if file + "_" + l1 + "_" + l2 not in dict.keys():
                dict[file + "_" + l1 + "_" + l2]=[np.mean(res_fpr)]
            else:
                dict[file + "_" + l1 + "_" + l2].append(np.mean(res_fpr))

            print(file + "_" + l1 + "_" + l2, np.mean(res_auroc))
            if file + "_" + l1 + "_" + l2 not in dict1.keys():
                dict1[file + "_" + l1 + "_" + l2] = [np.mean(res_auroc)]
            else:
                dict1[file + "_" + l1 + "_" + l2].append(np.mean(res_auroc))
            print('-------------------------------------------------------------------')


for key in dict.keys():
    print(dict)
    print(dict1)
    df = pd.DataFrame.from_dict(dict)
    df.to_excel("eval_open_ood/setup1_res_fpr.xls")
    print(key, np.mean(dict[key]), np.std(dict[key]))

    df = pd.DataFrame.from_dict(dict1)
    df.to_excel("eval_open_ood/setup1_res_auroc.xls")





