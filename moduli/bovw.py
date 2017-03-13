import numpy as np
from skimage import io as sio
from skimage.feature import daisy
from time import time
from matplotlib import pyplot as plt
from skimage.color import rgb2grey

def extract_features(dataset):
    nimgs = dataset.getLength()
    features = list()
    ni = 0 
    total_time = 0
    for cl in dataset.getClasses():
        paths = dataset.paths[cl]
        for impath in paths:
            t1 = time() 
            im = sio.imread(impath, as_grey = True) 
            feats = daisy(im, step = 4) 
            feats = feats.reshape((-1,200))
            features.append(feats) 
            t2 = time() 
            t3 = t2-t1
            total_time+=t3
            ni+=1 
            print "Image {0}/{1} [{2:0.2f}/{3:0.2f} sec]".format(ni,nimgs,t3,t3*(nimgs-ni))

    print "Stacking all features..."
    t1 = time()
    stacked = np.vstack(features)
    t2 = time()
    total_time+=t2-t2
    print "Total time: {0:0.2f} sec".format(total_time)
    return stacked

def extract_and_describe(img,kmeans):
    features = daisy(rgb2grey(img), step = 4).reshape((-1,200))
    assignments = kmeans.predict(features)
    histogram, _= np.histogram(assignments,bins=500,range=(0,499))
    return histogram

def display_image_and_representation(X,y,paths,classes,i):
	im = sio.imread(paths[i])
	plt.figure(figsize=(12,4))
	plt.suptitle("Class: {0} - Image: {1}".format(classes[y[i]],i))
	plt.subplot(1,2,1)
	plt.imshow(im)
	plt.subplot(1,2,2)
	plt.plot(X[i])
	plt.show()

def show_image_and_representation(img,image_representation):
    plt.figure(figsize=(13,4))
    plt.subplot(2,1,1)
    plt.imshow(img)
    plt.subplot(2,1,2)
    plt.plot(image_representation)
    plt.show()
	
def compare_representations(r1,r2):
	plt.figure(figsize=(12,4))
	plt.subplot(1,2,1)
	plt.plot(r1)
	plt.subplot(1,2,2)
	plt.plot(r2)
	plt.show()

def describe_dataset(dataset,kmeans):
    y = list() 
    X = list() 
    paths = list() 
    
    classes=dataset.getClasses()
    
    ni = 0
    t1 = time()
    for cl in classes:
        for path in dataset.paths[cl]: 
            img = sio.imread(path,as_grey = True)
            feat = extract_and_describe(img,kmeans)
            X.append(feat)
            y.append(classes.index(cl)) 
            paths.append(path) 
            ni+= 1

    X = np.array(X)
    y = np.array(y)
    t2 = time()
    print "Elapsed time {0:0.2f}".format(t2-t1)
    return X,y,paths