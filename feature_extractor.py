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

dataset_food = Dataset(dataset_dir)
training_set, test_set = dataset_food.splitTrainingTest(0.7)

print "Number of images in the training set: {0}".format(training_set.getLength())
print "Number of images in the test set: {0}".format(test_set.getLength())

print "Total number of images: {0}".format(training_set.getLength() + test_set.getLength())

warnings.filterwarnings("ignore")

training_local_features = extract_features(training_set)

kmeans = KMeans(500)
kmeans.fit(training_local_features)
training_features, y_training, paths_training = describe_dataset(training_set, kmeans)
test_features, y_test, paths_test = describe_dataset(test_set, kmeans)

norm = Normalizer(norm='l2')
training_features = norm.transform(training_features)
test_features = norm.transform(test_features)

X_training = training_features
y_training = y_training
X_test = test_features
y_test = y_test

with open(dataset_dir + "features.pkl","wb") as f:
	cPickle.dump({
		"X_training" : X_training,
		"y_training" : y_training,
		"X_test" : X_test,
		"y_test" : y_test,
		"paths_training": paths_training,
		"paths_test": paths_test
		},f)

#Caricamento
'''
with open(dataset_dir + "UNICT-FlickrFood/features.pkl") as f:
	data = cPickle.load(f)
	X_training = data["X_training"]
	y_training = data["y_training"]
	X_test = data["X_test"]
	y_test = data["y_test"]
	paths_training = data["paths_training"]
	paths_test = data["paths_test"]

print data	
'''