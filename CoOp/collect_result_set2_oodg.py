import os
import numpy as np
import linecache
import pandas as pd
import xlsxwriter

PATH = "output/CRoFT/"
plist = os.listdir(PATH)
#print(plist)
plists = []
for t in plist:
#    if "PACS" in t:
#        domains = ['test_on_photo.json', 'test_on_cartoon.json',  'test_on_art_painting.json',  'test_on_sketch.json']
#        plists.append(t)
    if "VLCS" in t:
        domains = ["test_on_pascal.json",  "test_on_caltech.json",  "test_on_sun.json",  "test_on_labelme.json"]
        plists.append(t)


for file in plists:
    print(file)
    ACC_all = {}
    for domain in domains:
        root = 'output/CRoFT/' + file + "/" + domain
        key = domain
        ACC = {}
        P0 = os.listdir(root)
        for p0 in P0:
            key0 = p0[-1]
            ACC[key0] = {}
            path0 = os.path.join(root, p0)
            P1 = os.listdir(path0)
            for p1 in P1:
                key1 = p1
                ACC[key0][key1] = []
                path1 = os.path.join(path0, p1)
                log_path = os.path.join(path1, "log.txt")

                line = linecache.getline(log_path, 100)
                i = 1
                while 'Deploy the model with the best val performance' not in line:
                    line = linecache.getline(log_path, 100 + i)
                    i += 1
                    if i > 50000:
                        break
                for j in range(16):
                    get_epoch = linecache.getline(log_path, 100 + i + j)
                    if 'epoch' in get_epoch:
                        epoch = int(get_epoch.split('(')[-1].split(')')[0].split('=')[-1].strip(' '))
                        break

                with open(log_path) as f:
                    for idx in range(30):
                        acc_line = linecache.getline(log_path, 100 + i + idx)
                        if 'Do evaluation on test set' in acc_line:
                            acc_line = linecache.getline(log_path, 100 + i + idx + 4)
                            # print(float(acc_line.split(':')[-1].split('%')[0].strip()))
                            ACC[key0][key1] = [float(acc_line.split(':')[-1].split('%')[0].strip())]
                            break
                        idx += 1

                # get best model's val_acc (few-shots of val set)
                with open(log_path) as f:
                    idx = 1
                    for l in f.readlines():
                        if ('epoch [' + str(epoch)) in l:
                            for k in range(20):
                                acc_line = linecache.getline(log_path, idx + k)
                                if 'Do evaluation on val set' in acc_line:
                                    acc_line = linecache.getline(log_path, idx + k + 4)
                                    ACC[key0][key1].append(float(acc_line.split(':')[-1].split('%')[0].strip()))
                                    break
                            break
                        idx += 1
                # get model's test_acc at the last training step
                with open(log_path) as f:
                    idx = 1
                    for l in f.readlines():
                        if ('epoch [' + str(30)) in l:
                            for k in range(20):
                                acc_line = linecache.getline(log_path, idx + k)
                                if 'Do evaluation on test set' in acc_line:
                                    acc_line = linecache.getline(log_path, idx + k + 4)
                                    ACC[key0][key1].append(float(acc_line.split(':')[-1].split('%')[0].strip()))
                                    break
                            break
                        idx += 1
        ACC_all[key] = ACC

    print(ACC_all)
    test_acc = []
    val_acc = []
    for domain in ACC_all.keys():
        for seed in ACC_all[domain].keys():
            if len(ACC_all[domain][seed]['shots16']) >= 1:
                test_acc.append(ACC_all[domain][seed]['shots16'][0])
                val_acc.append(ACC_all[domain][seed]['shots16'][1])
            else:
                test_acc.append(0)
                val_acc.append(0)
    test_seed1, test_seed2, test_seed3 = 0, 0, 0
    val_seed1, val_seed2, val_seed3 = 0, 0, 0
    print(test_acc)

    for n in range(len(test_acc)):
        if n % 3 == 0:
            test_seed1 += test_acc[n]
            val_seed1 += val_acc[n]
        if n % 3 == 1:
            test_seed2 += test_acc[n]
            val_seed2 += val_acc[n]
        if n % 3 == 2:
            test_seed3 += test_acc[n]
            val_seed3 += val_acc[n]

    print('results_seed1:', test_seed1 / len(domains), val_seed1 / 4)
    print('results_seed2:', test_seed2 / len(domains), val_seed2 / 4)
    print('results_seed3:', test_seed3 / len(domains), val_seed3 / 4)
    print('results:', (test_seed1 + test_seed2 + test_seed3) / len(domains) / 3)

