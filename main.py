# -*- coding: iso-8859-15 -*-
import sys
import cPickle
import warnings

import numpy as np 

from matplotlib import pyplot as plt 
from skimage import io as sio 
from skimage.feature import daisy

sys.path.append('moduli/')
from dataset import Dataset

from bovw import extract_features
from bovw import describe_dataset

from sklearn.cluster import MiniBatchKMeans as KMeans
from sklearn.preprocessing import Normalizer

dataset_dir = 'dataset/'
warnings.filterwarnings("ignore")

#Caricamento

with open(dataset_dir + "features.pkl") as f:
	data = cPickle.load(f)
	X_training = data["X_training"]
	y_training = data["y_training"]
	X_test = data["X_test"]
	y_test = data["y_test"]
	paths_training = data["paths_training"]
	paths_test = data["paths_test"]

#print data	

def display_img_and_representation(x, y, pathimage, y_etichetta):

	
	print y[y_etichetta]

	img = sio.imread(pathimage)
	
	plt.figure(figsize=(12,4))

	plt.subplot(1,2,1)
	plt.imshow(img)

	plt.subplot(1,2,2)
	plt.plot(x)

	plt.show()

from sklearn.neighbors import KDTree
tree = KDTree(X_training)

def query_image(tree, paths_training, index_training, paths_test, index_test, x_test):
	query_im = sio.imread(paths_test[index_test])
	closest_im = sio.imread(paths_training[index_training])

	plt.figure(figsize=(12,4))
	plt.subplot(1,2,1)
	plt.imshow(query_im)
	plt.title("Query image {0}".format(paths_test[index_test]))

	plt.subplot(1,2,2)
	plt.imshow(closest_im)
	plt.title("Closest Image {0}".format(paths_training[index_training]))

	plt.show()

index_test = np.random.randint(len(X_test))

query_feature = X_test[index_test]
distance, index = tree.query(query_feature)

print distance[0,0]
index_training = index[0,0]
print index_training

def nearestimage_and_representation(paths_training, index_training, X_training, y_training, paths_test, index_test, X_test, y_test, distance, accuracy, matrix):
	plt.figure()

	query_image = sio.imread(paths_test[index_test])
	plt.subplot2grid((2,3), (0, 0))
	if(y_test[index_test] == 0):
		typeofclass = "Food"
	else:
		typeofclass = "Not-Food"
	plt.title("Query image {0}".format(paths_test[index_test]) )
	plt.imshow(query_image)

	plt.subplot2grid((2,3), (0, 1))
	plt.title("Class {0}".format(typeofclass) )
	plt.plot(X_test[index_test])

	closest_im = sio.imread(paths_training[index_training])
	plt.subplot2grid((2,3), (1, 0))
	if(y_training[index_training] == 0):
		typeofclass = "Food"
	else:
		typeofclass = "Not-Food"
	plt.title("Closest Im {0}".format(paths_training[index_training]) )
	plt.imshow(closest_im)

	plt.subplot2grid((2,3), (1, 1))
	plt.title("Class {0}".format(typeofclass) )
	plt.plot(X_training[index_training])
	
	plt.subplot2grid((2,3), (0, 2), rowspan=2)
	plt.title("Distance {0} - Accuracy {1}%\nConfusion Matrix\n {2}".format(distance, accuracy, matrix))
	plt.plot(X_test[index_test], color='red')
	plt.plot(X_training[index_training], color='blue')
	
	plt.show()

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score, confusion_matrix

nn1 = KNN(1)
nn1.fit(X_training, y_training)
predicted_labels = nn1.predict(X_test)
a = accuracy_score(y_test, predicted_labels)
M = confusion_matrix(y_test, predicted_labels)

'''
print "\n1‐NN, accuracy: %0.2f, Confusion Matrix:\n" %a
print "Accuracy: %0.2f" %nn1.score(X_test, y_test)
print M
'''

nearestimage_and_representation(paths_training, index_training, X_training, y_training, paths_test, index_test, X_test, y_test, round(distance[0,0],2), round(a,2), M)

'''
print y_training.shape
print y_test.shape

'''
nn1 = KNN(1)
nn1.fit(X_training, y_training)
predicted_labels = nn1.predict(X_test)
a = accuracy_score(y_test, predicted_labels)
M = confusion_matrix(y_test, predicted_labels)

print "\n1‐NN, accuracy: %0.2f, Confusion Matrix:\n" %a
print "Accuracy: %0.2f" %nn1.score(X_test, y_test)
print M

nn5 = KNN(5)
nn5.fit(X_training, y_training)
predicted_labels = nn5.predict(X_test)
a = accuracy_score(y_test, predicted_labels)
M = confusion_matrix(y_test, predicted_labels)

print "\n5‐NN, accuracy: %0.2f, Confusion Matrix:\n" %a
print "Accuracy: %0.2f" %nn5.score(X_test, y_test)
print M

nn51 = KNN(51)
nn51.fit(X_training, y_training)
predicted_labels = nn51.predict(X_test)
a = accuracy_score(y_test, predicted_labels)
M = confusion_matrix(y_test, predicted_labels)

print "\n51‐NN, accuracy: %0.2f, Confusion Matrix:\n" %a
print "Accuracy: %0.2f" %nn51.score(X_test, y_test)
print M

