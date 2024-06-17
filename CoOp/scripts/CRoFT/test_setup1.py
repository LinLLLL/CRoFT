import os
import numpy as np
import random


for lambda2 in [1, 5, 10, 15, 20]:
    for lambda1 in [1, 5, 10, 15, 20]: 
        os.system("bash test_imagenet.sh " + str(lambda1) + ' ' + str(lambda2) + ' 0' + ' 1')
        os.system("bash test_imagenet.sh " + str(lambda1) + ' ' + str(lambda2) + ' 0' + ' 16')
        os.system("bash test_imagenet.sh " + str(lambda1) + ' ' + str(lambda2) + ' 0' + ' 32')


for lambda2 in [1, 5, 10, 15, 20]:
    for lambda1 in [1, 5, 10, 15, 20]: 
        os.system("bash test_openood_setup1.sh " + str(lambda1) + ' ' + str(lambda2) + ' 0' + ' 1')
        os.system("bash test_openood_setup1.sh " + str(lambda1) + ' ' + str(lambda2) + ' 0' + ' 16')
        os.system("bash test_openood_setup1.sh " + str(lambda1) + ' ' + str(lambda2) + ' 0' + ' 32')


