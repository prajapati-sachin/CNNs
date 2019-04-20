import numpy as np
import cv2
import glob
import pylab as plt
import csv
from sklearn.decomposition import PCA, IncrementalPCA
from itertools import combinations
import random
from joblib import dump, load
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

folders50 = sorted(glob.glob("./train50/*"))
foldersval = sorted(glob.glob("./validation/*"))
foldersfull = sorted(glob.glob("./trainfull/*"))


##########################################################################
# FOR PCA
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
# print(X50.shape)
# # pca = PCA(n_components=50)
# pca = IncrementalPCA(n_components=50, batch_size=100)
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
#     temp = temp.reshape(50)
#     Xfull.append(list(temp))

# Xfull = np.array(Xfull)

# np.save("dataXfull", Xfull)

# ########### Input Y #########
# Y = []

# for rewards in reward_list:
# 	temp = []
# 	with open(rewards) as fileX:
# 		x_reader = csv.reader(fileX)
# 		for row in x_reader:
# 			temp.append(int(float(row[0])))
# 		Y.append(temp)

# Y = np.array(Y)
# np.save("dataYfull", Y)
# #############################

# Loading
# Xfull = np.load("dataXfull.npy")
# print(Xfull.shape)

# Y = np.load("dataYfull.npy")
# print(Y.shape)

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

Xtrain_pos = []
Ytrain_pos = []

Xtrain_neg = []
Ytrain_neg = []


Yindexing = [None]*len(Y)
Yindexing[0] = 0

for i in range(1, len(Y)):
	Yindexing[i] = Yindexing[i-1] + len(Y[i-1])+1

print(Yindexing)


for i in range(len(Yindexing)):
	num_images = len(Y[i])+1
	for j in range(7, len(Y[i])):
		if(Y[i][j]==1):
			combs = getcomb(j)
			for k in range(len(combs)):
				templist = combs[k] 			
				templist.append(j-1)
				# print(templist)
				temptrain = []
				for l in range(len(templist)):
					final_index = Yindexing[i] + templist[l]
					temptrain += list(Xfull[final_index])
				Xtrain_pos.append(temptrain)
				Ytrain_neg.append(1)
		else:
			if(decision(0.15)):
				combs = getcomb(j)
				for k in range(len(combs)):
					if(decision(0.5)):
						templist = combs[k] 			
						templist.append(j-1)
						# print(templist)
						temptrain = []
						for l in range(len(templist)):
							final_index = Yindexing[i] + templist[l]
							temptrain += list(Xfull[final_index])
						Xtrain_neg.append(temptrain)
						Ytrain_neg.append(Y[i][j])



Xtrain = Xtrain_pos[0:10000] + Xtrain_neg[0:10000]
Ytrain = Ytrain_pos[0:10000] + Ytrain_neg[0:10000]
np.save("dataXtrain", Xtrain)
np.save("dataYtrain", Ytrain)

Xtrain = np.load("dataXtrain.npy")
Ytrain = np.load("dataYtrain.npy")
print(Xtrain.shape)
print(Ytrain.shape)





Xtrain_pos = np.array(Xtrain_pos)
Ytrain_pos = np.array(Ytrain_pos)
Xtrain_neg = np.array(Xtrain_neg)
Ytrain_neg = np.array(Ytrain_neg)


np.save("dataXtrain_pos", Xtrain_pos)
np.save("dataYtrain_pos", Ytrain_pos)
np.save("dataXtrain_neg", Xtrain_neg)
np.save("dataYtrain_neg", Ytrain_neg)



#Loading
Xtrain_pos = np.load("dataXtrain_pos.npy")
Xtrain_neg = np.load("dataXtrain_neg.npy")
print(Xtrain_pos.shape)
print(Xtrain_neg.shape)



Ytrain_pos = np.load("dataYtrain_pos.npy")
Ytrain_neg = np.load("dataYtrain_neg.npy")
print(Ytrain_pos.shape)
print(Ytrain_neg.shape)






##########################################################################
#TRAINING SVM

pos = 0
neg = 0
for i in range(len(Ytrain)):
	if(Ytrain[i]==0):
		neg+=1
	elif(Ytrain[i]==1):
		pos+=1
	else:
		print("Problem")



print("Postive in train: ", pos)
print("Negative in train: ", neg)
print("Total in train: ", len(Ytrain))

clf = svm.SVC(gamma='auto', kernel='linear')
clf.fit(Xtrain, Ytrain)

##########################################################################
#TAKING INPUT OF VALIDATION SET

valX = []        
valY = []

c=0
for folder in foldersval:
	c+=1
	if(c%1000==0):
		print("Input of Validation data: ", c)
	imagesval_list = []
	reading_images = []
	tempo = []
	for f in sorted(glob.glob(folder+"/*.png")):
		imagesval_list.append(f)

	for image in imagesval_list:
		img  = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
		reading_images.append(img)

	for i in range(len(reading_images)):
		temp = np.array(reading_images[i])
		tempimg = temp.flatten()
		tempimg = tempimg.reshape(33600, 1)
		tempimg = tempimg.transpose()
		tempimg = pca.transform(tempimg)
		tempimg = tempimg.reshape(50)
		tempo += list(tempimg)

	# if(len(tempo)!=5):
	# 	print(folder)
	# print(len(tempo))
	valX.append(tempo)


with open("./rewardsval.csv") as fileX:
	x_reader = csv.reader(fileX)
	for row in x_reader:
		valY.append(int(float(row[1])))

valX = np.array(valX)
valY = np.array(valY)

np.save("datavalX", valX)
np.save("datavalY", valY)

#Loading
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



print("Postive in val: ", pos)
print("Negative in val: ", neg)
print("Total in val: ", len(valY))



##########################################################################
#TESTING ON VALIDATION DATA

Ypredval = clf.predict(valX)

accuracy = accuracy_score(valY, Ypredval)
print("Accuracy: ", accuracy)

f1 = f1_score(valY, Ypredval, average=None) 
print(f1)

confusion = confusion_matrix(valY, Ypredval)
print(confusion)
