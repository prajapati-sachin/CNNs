import sys
import csv
import math
import time
import numpy as np
import scipy
import cv2
import glob
import random



BATCHSIZE = 128
curbatch = 865

for i in range(1, 6):
	print("Processing file: ", i)
	file1 = "set2/Xtrain15_" + str(i) + ".npy"
	file2 = "set2/Ytrain15_" + str(i) + ".npy"
	X = np.load(file1)
	Y = np.load(file2)
	num_examples = (Y.shape)[0]
	num_batches = num_examples/BATCHSIZE
	print("Num of batches in ", file1, "=", int(num_batches))
	for j in range(int(num_batches)):
		print("Making batch no.: ", j)
		tempX = X[j*BATCHSIZE:(j*BATCHSIZE)+BATCHSIZE]
		tempY = Y[j*BATCHSIZE:(j*BATCHSIZE)+BATCHSIZE]
		tempfile1 = "batchesfull/X_" + str(curbatch)
		tempfile2 =	"batchesfull/Y_" + str(curbatch)
		np.save(tempfile1, tempX)
		np.save(tempfile2, tempY)
		curbatch+=1


print("Total batches made: ", curbatch-1)