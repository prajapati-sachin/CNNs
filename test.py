import numpy as np
import cv2
import glob
import pylab as plt
import csv
from sklearn.decomposition import PCA
from itertools import combinations
import random
from joblib import dump, load
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.models import load_model


valX = np.load("datavalX.npy")
print(valX.shape)

valY = np.load("datavalY.npy")
print(valY.shape)


pos = 0
neg = 0
for i in range(len(valY)):
	if(valY[i]==0):
		neg+=1
	elif(valY[i]==1):
		pos+=1
	else:
		print("Problem")


model = load_model("firstmodelbatch1")

Ypred = model.predict_classes(valX)

# print(type(Ypred))

accuracy = accuracy_score(valY, Ypred)
print("Accuracy: ", accuracy)

f1 = f1_score(valY, Ypred, average=None) 
print(f1)

print("F1 Score: ", np.mean(f1))

confusion = confusion_matrix(valY, Ypred)
print(confusion)
