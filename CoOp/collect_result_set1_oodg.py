import os
import numpy as np
import pandas as pd

path = 'output/CRoFT'
files = os.listdir(path)


dict1 = {}
for seed in range(1):
    for file in files:
        if "shot" not in file:
            continue
        shots_path = os.path.join(path, file)
        shot_files = os.listdir(shots_path)
        acc = {}
        acc['CRoFT'] = []  #
        l1 = -1
        for log in shot_files:
            file_path = os.path.join(shots_path, log)
            # print(file_path)
            with open(file_path, 'r') as f:
                i = 0
                c = 100000
                for line in f.readlines():
                    i += 1

                    if 'Loading weights to generator from' in line:
                        seed = line.split("seed")[-1].split("/")[0].strip()
                        # print(line.split("imagenet46")[-1].split("_"))
                        l1 = line.split("imagenet46")[-1].split("_")[1].strip()
                        l2 = line.split("imagenet46")[-1].split("_")[2].split("/seed")[0].strip()
                    if '=> result' in line:
                        c = i
                    if i == c + 3:
                        A = np.float32(line.split(': ')[-1].strip('%\n'))
                        c = 100000
                # acc['CRoFT'].append(A)
                if l1 + "_" + l2 in acc.keys():
                    acc[l1 + "_" + l2].append(A)
                else:
                    acc[l1 + "_" + l2]=[A]

        print(acc)   
        print('-------------------------------------------------------------------')
        for key in acc.keys():
            dict1[key+'_'+str(file)] = [np.mean(acc[key]), np.std(acc[key])]

        print('-------------------------------------------------------------------')
print(dict1)

for key in dict1.keys():
    df = pd.DataFrame.from_dict(dict1)
    df.to_excel("output/CRoFT-OODG-ACC.xls")






