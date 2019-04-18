import numpy as np
import cv2
import glob
import pylab as plt
import csv
from sklearn.decomposition import PCA
from itertools import combinations
import random


folders50 = sorted(glob.glob("./train50/*"))
foldersfull = sorted(glob.glob("./trainfull/*"))
foldersval = sorted(glob.glob("./validation/*"))

# # print(len(folders50))
# # print(len(foldersfull))
# # print((folders))

images_list = []
# image_num = []
reward_list = []


for folder in foldersfull:
    for f in sorted(glob.glob(folder+"/*.png")):
        images_list.append(f)
    reward_list.append(folder+"/rew.csv")
	
read_images = []        


# # print(len(images_list))
# # print((images_list[0:10]))
# # print((reward_list))
# # print(image_num)

c=0
for image in images_list:
    c+=1
    if(c%2000==0):
    	print(c/2000)
    # print(c)
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (105, 80))
    read_images.append(img)

X = []

for i in range(len(read_images)):
	temp = np.array(read_images[i])	
	X.append(temp.flatten())


X = np.array(X)
print(X.shape)
np.save("dataX", X)

Y = []

for rewards in reward_list:
	temp = []
	with open(rewards) as fileX:
		x_reader = csv.reader(fileX)
		for row in x_reader:
			temp.append(int(float(row[0])))
		Y.append(temp)

Y = np.array(Y)
# print(len(Y[0]))

np.save("dataY", Y)


X = np.load("dataX.npy")
Y = np.load("dataY.npy")

print(X.shape)
print(Y.shape)

num=50
for i in range(50):
	num+=len(Y[i])


print(num)

pca = PCA(n_components=50)
pca.fit(X[0:num])

Xnew = pca.transform(X)

np.save("dataXnew", Xnew)


Xnew = np.load("dataXnew.npy")
print(Xnew.shape)

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


Xtrain = []
Ytrain = []


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
					temptrain += list(Xnew[final_index])
				Xtrain.append(temptrain)
				Ytrain.append([1])
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
							temptrain += list(Xnew[final_index])
						Xtrain.append(temptrain)
						Ytrain.append([Y[i][j]])




Xtrain = np.array(Xtrain)
Ytrain = np.array(Ytrain)
print(Xtrain.shape)
print(Ytrain.shape)


np.save("Xtrain", Xtrain)
np.save("Ytrain", Ytrain)

Xtrain = np.load("Xtrain.npy")
Ytrain = np.load("Ytrain.npy")
print(Xtrain.shape)
print(Ytrain.shape)


# print(type(getcomb(7)[0]))
# print(list(getcomb(7)[0]))
# print(type(list(getcomb(7)[0])))


valX = []        
valY = []

for folder in foldersval:
	imagesval_list = []
	reading_images = []
    for f in sorted(glob.glob(folder+"/*.png")):
        imagesval_list.append(f)
	for image in images_list:
	    read_images.append(cv2.imread(image, cv2.IMREAD_GRAYSCALE))
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
	    img = cv2.resize(img, (105, 80))
    	reading_images.append(img)

    tempo = []

	for i in range(len(reading_images)):
		temp = np.array(reading_images[i])	
		tempimg = temp.flatten()
		tempimg = pca.transform(tempimg)	    
		tempo += list(tempimg)

	valX.append(tempo)



with open("./validation/rewards.csv") as fileX:
	x_reader = csv.reader(fileX)
	for row in x_reader:
		valY.append(int(float(row[0])))

valX = np.array(valX)
valY = np.array(valY)


print(valX.shape)
print(valY.shape)
