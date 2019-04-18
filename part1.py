import numpy as np
import cv2
import glob
import pylab as plt
import csv
from sklearn.decomposition import PCA

folders50 = sorted(glob.glob("./train50/*"))
foldersfull = sorted(glob.glob("./trainfull/*"))

# print(len(folders50))
# print(len(foldersfull))
# print((folders))

images_list = []
# image_num = []
reward_list = []

for folder in foldersfull:
    for f in sorted(glob.glob(folder+"/*.png")):
        images_list.append(f)
    reward_list.append(folder+"/rew.csv")

read_images = []        

# print(len(images_list))
# print((images_list[0:10]))
# print((reward_list))
# print(image_num)

c=0
for image in images_list:
    c+=1
    if(c%2000==0):
    	print(c/2000)
    # print(c)
    read_images.append(cv2.imread(image, cv2.IMREAD_GRAYSCALE))
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
print(len(Y[1]))

np.save("dataY", Y)


# X = np.load("dataX.npy")
# Y = np.load("dataY.npy")

# print(X.shape)
# print(Y)

# pca = PCA(n_components=50)
# Xnew = pca.fit_transform(X)

# np.save("dataXnew", Xnew)

# Xnew = np.load("dataXnew.npy")
# print(Xnew.shape)
