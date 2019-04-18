import numpy as np
import cv2
import glob
import pylab as plt
import csv
from sklearn.decomposition import PCA
from itertools import combinations
import random
from joblib import dump, load


folders50 = sorted(glob.glob("./train2/*"))
foldersval = sorted(glob.glob("./validation/*"))
foldersfull = sorted(glob.glob("./train5/*"))


##########################################################################
# # FOR PCA
# images_list50 = []

# for folder in folders50:
#     for f in sorted(glob.glob(folder+"/*.png")):
#         images_list50.append(f)



# read_images50 = []        
# c=0
# for image in images_list50:
#     c+=1
#     if(c%2000==0):
#     	print("Input of PCA data: ", c/2000)
#     # print(c)
#     read_images50.append(cv2.imread(image, cv2.IMREAD_GRAYSCALE))


# X50 = []

# for i in range(len(read_images50)):
# 	temp = np.array(read_images50[i])	
# 	X50.append(temp.flatten())

# X50 = np.array(X50)

# pca = PCA(n_components=50)
# pca.fit(X50)
# dump(pca, 'pca50.joblib')

#Loading
pca = load('pca50.joblib')

##########################################################################
#TAKING INPUT OF ALL IMAGES AND REDUCING DIMENSION AT SAME TIME

# images_list = []
# reward_list = []
# for folder in foldersfull:
#     for f in sorted(glob.glob(folder+"/*.png")):
#         images_list.append(f)
#     reward_list.append(folder+"/rew.csv")

# Xfull = []

# # read_images = []        
# c=0
# for image in images_list:
#     c+=1
#     if(c%2000==0):
#     	print("Input of full data: ", c/2000)
#     img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
#     temp = np.array(img)
#     temp = temp.flatten()
#     temp = temp.reshape(33600, 1)
#     temp = temp.transpose()
#     temp = pca.transform(temp)
#     temp =temp.reshape(50)
#     Xfull.append(list(temp))

# Xfull = np.array(Xfull)

# np.save("dataXfull", Xfull)

# ########### Input Y #########
# Yfull = []

# for rewards in reward_list:
# 	temp = []
# 	with open(rewards) as fileX:
# 		x_reader = csv.reader(fileX)
# 		for row in x_reader:
# 			temp.append(int(float(row[0])))
# 		Yfull.append(temp)

# Yfull = np.array(Yfull)
# np.save("dataYfull", Yfull)
# #############################

#Loading
Xfull = np.load("dataXfull.npy")
print(Xfull.shape)

Yfull = np.load("dataYfull.npy")
print(Yfull.shape)

##########################################################################
#EXTRA FUNCTIONS

def decision(probability):
	return random.random() < probability

def tolist(t):
	temp = [] 
	for i in range(4):
		temp.append(t[i])
	return temp


def getcomb(j):
	result = []
	a = j-7
	b = j-2
	temp = range(a, b+1)
	temp1 = list(combinations(temp, 4))
	for i in range(len(temp1)):
		result.append(list(temp1[i]))
	return result

##########################################################################
#GENERATING TRAINING DATA

