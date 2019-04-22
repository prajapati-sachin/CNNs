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
# pca = load('pca50.joblib')

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
#     img = cv2.imread(image)
#     temp = np.array(img)
#     # print(temp.shape)
#     # temp = temp.flatten()
#     # temp = temp.reshape(33600, 1)
#     # temp = temp.transpose()
#     # temp = pca.transform(temp)
#     # temp = temp.reshape(50)
#     Xfull.append(list(temp))

# Xfull = np.array(Xfull)
# print(Xfull.shape)

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

#Loading
# Xfull = np.load("../sachin/xc15.npy")
# print(Xfull.shape)

# Y = np.load("../sachin/yc15.npy")
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

def f1score(ytrue, ypred):
	return np.mean(f1_score(ytrue, ypred, average=None))



##########################################################################
#GENERATING TRAINING DATA

# Xtrain = []
# Ytrain = []


# Yindexing = [None]*len(Y)
# Yindexing[0] = 0

# for i in range(1, len(Y)):
# 	Yindexing[i] = Yindexing[i-1] + len(Y[i-1])+1

# # print(Yindexing)


# for i in range(len(Yindexing)):
# 	num_images = len(Y[i])+1
# 	for j in range(7, len(Y[i])):
# 		if(Y[i][j]==1):
# 			if(decision(0.4)):
# 				combs = getcomb(j)
# 				for k in range(len(combs)):
# 					if(decision(0.333)):
# 						templist = combs[k] 			
# 						templist.append(j-1)
# 						# print(templist)
# 						# temptrain = []
# 						temptrain = Xfull[Yindexing[i] + templist[0]]						
# 						# temptrain = np.empty([210, 160, 3])
# 						for l in range(1, len(templist)):
# 							final_index = Yindexing[i] + templist[l]
# 							# temptrain.append(list(Xfull[final_index]))
# 							# print(temptrain.shape)
# 							# print(Xfull[final_index].shape)
# 							temptrain = np.append(temptrain, Xfull[final_index], axis=2)
# 							# print(temptrain.shape)
# 						# print(temptrain.shape)
# 						Xtrain.append(list(temptrain))
# 						Ytrain.append(1)
# 		else:
# 			if(decision(0.012)):
# 				combs = getcomb(j)
# 				for k in range(len(combs)):
# 					if(decision(0.333)):
# 						templist = combs[k] 			
# 						templist.append(j-1)
# 						# print(templist)
# 						# temptrain = []
# 						temptrain = Xfull[Yindexing[i] + templist[0]]
# 						# print(temptrain.shape)
# 						# print(type(temptrain))
# 						# temptrain = np.empty([210, 160, 3])
# 						for l in range(1, len(templist)):
# 							final_index = Yindexing[i] + templist[l]
# 							# temptrain.append(list(Xfull[final_index]))
# 							# print(temptrain.shape)
# 							# print(Xfull[final_index].shape)
# 							temptrain = np.append(temptrain, Xfull[final_index], axis=2)
# 							# print(temptrain.shape)							
# 						# print(temptrain.shape)
# 						Xtrain.append(list(temptrain))
# 						Ytrain.append(Y[i][j])




# Xtrain = np.array(Xtrain)
# Ytrain = np.array(Ytrain)

# np.save("Xtrain15_5", Xtrain)
# np.save("Ytrain15_5", Ytrain)

# # Loading
# # Xtrain = np.load("Xtrain_2.npy")
# print(Xtrain.shape)

# # Ytrain = np.load("Ytrain_2.npy")
# print(Ytrain.shape)

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

##########################################################################
#TRAINING SVM

# clf = svm.SVC(gamma='auto', kernel='rbf')
# clf.fit(Xtrain, Ytrain)

##########################################################################
#TAKING INPUT OF VALIDATION SET

# valX = []        
# valY = []

# c=0
# for folder in foldersval:
# 	c+=1
# 	if(c%1000==0):
# 		print("Input of Validation data: ", c)
# 	imagesval_list = []
# 	reading_images = []
# 	for f in sorted(glob.glob(folder+"/*.png")):
# 		imagesval_list.append(f)

# 	for image in imagesval_list:
# 		img  = cv2.imread(image)
# 		reading_images.append(img)

# 	# tempo = []
# 	tempo = np.array(reading_images[0])
# 	for i in range(1, len(reading_images)):
# 		temp = np.array(reading_images[i])
# 		# tempo.append(list(temp))
# 		tempo = np.append(tempo, temp, axis=2)

# 	# if(len(tempo)!=5):
# 	# 	print(folder)
# 	# print(len(tempo))
# 	# print(tempo.shape)
# 	valX.append(list(tempo))


# with open("./rewardsval.csv") as fileX:
# 	x_reader = csv.reader(fileX)
# 	for row in x_reader:
# 		valY.append(int(float(row[1])))

# valX = np.array(valX)
# valY = np.array(valY)

# np.save("datavalX", valX)
# np.save("datavalY", valY)

# # Loading
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



# print("Postive in val: ", pos)
# print("Negative in val: ", neg)
# print("Total in val: ", len(valY))



##########################################################################
#TESTING ON VALIDATION DATA

# Ypredval = clf.predict(valX)

# accuracy = accuracy_score(valY, Ypredval)
# print("Accuracy: ", accuracy)

# f1 = f1_score(valY, Ypredval, average=None) 
# print(f1)

# confusion = confusion_matrix(valY, Ypredval)
# print(confusion)
##########################################################################
#DATA GENERATOR

num_batches = 864
batch_size = 128
# num_batches = 864

class DataGenrator(keras.utils.Sequence):
	def __init__(self,  dim=(210, 160 ,15), batch_size =128 , shuffle=True):
		self.dim = dim
		self.shuffle = shuffle
		self.on_epoch_end()

	def __len__(self):
		return num_batches

	def __getitem__(self, index):
		x = np.load("batches1/X_" + str(index) + ".npy")
		y = np.load("batches1/Y_" + str(index) + ".npy")
		return x, y

	def on_epoch_end(self):
		self.indexes = np.arange(num_batches * batch_size)
		if(self.shuffle):
			np.random.shuffle(self.indexes)






##########################################################################
#MAKING THE ARCHITECTURE


#create model
# model = Sequential()

# #add model layers
# # model.add(Conv3D(32, kernel_size=3, activation=’relu’, input_shape=(28,28,1)))

# model.add(Conv3D(32, kernel_size = (3, 3, 15), activation=’relu’, strides=(2, 2), input_shape=(210, 160, 15)))
# model.add(MaxPooling3D(pool_size=(2, 2, 15), strides=(2, 2)))

# model.add(Conv3D(64, kernel_size = (3, 3, 15), activation=’relu’,  strides=(2, 2), input_shape=(210, 160, 15)))
# model.add(MaxPooling3D(pool_size=(2, 2, 15), strides=(2, 2)))

# model.add(Dense(1024))


# #compile model using accuracy to measure model performance
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



#Manish model

# def get_model():
model = Sequential()
model.add(Conv2D(32, (3,3), strides=2, activation='relu', input_shape = (210,160,15)))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))

model.add(Conv2D(64, (3,3), strides=2, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))

model.add(Flatten())
model.add(Dense(2048, activation='relu'))

model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# 	return model

training_generator = DataGenrator(128)

epochs = 20

model.fit_generator(generator=training_generator, epochs = epochs, validation_data=(valX, valY), use_multiprocessing=True, workers=25)

