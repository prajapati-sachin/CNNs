import sys
import csv
import math
import time
import numpy as np
import scipy
import cv2
import glob
import random

Xtrain = np.load("batches1/X_864.npy")
print(Xtrain.shape)

Ytrain = np.load("batches1/Y_864.npy")
print(Ytrain.shape)

# pos = 0
# neg = 0
# for i in range(len(Ytrain)):
# 	if(Ytrain[i]==0):
# 		neg+=1
# 	elif(Ytrain[i]==1):
# 		pos+=1
# 	else:
# 		print("Problem")



# print("Postive in train: ", pos)
# print("Negative in train: ", neg)
# print("Total in train: ", len(Ytrain))
