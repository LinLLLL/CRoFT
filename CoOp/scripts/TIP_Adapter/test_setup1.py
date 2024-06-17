import os
import numpy as np
import random

os.system("bash test_imagenet.sh " + ' 0' + ' ' + str(1) + ' ' + str(0.5) + ' ' + str(5.5))
os.system("bash test_imagenet.sh " + ' 0' + ' ' + str(16) + ' ' + str(0.5) + ' ' + str(5.5))
os.system("bash test_imagenet.sh " + ' 0' + ' ' + str(32) + ' ' + str(0.5) + ' ' + str(5.5))

os.system("bash test_openood_setup1.sh " + ' 0' + ' ' + str(1) + ' ' + str(0.5) + ' ' + str(5.5))
os.system("bash test_openood_setup1.sh " + ' 0' + ' ' + str(16) + ' ' + str(0.5) + ' ' + str(5.5))
os.system("bash test_openood_setup1.sh " + ' 0' + ' ' + str(32) + ' ' + str(0.5) + ' ' + str(5.5))