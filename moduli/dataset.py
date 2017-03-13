import os
import glob
from skimage import io as sio
from matplotlib import pyplot as plt
import numpy as np
from copy import copy

class Dataset:
    def __init__(self,path_to_dataset):
        self.path_to_dataset = path_to_dataset
        classes = os.listdir(path_to_dataset)
        self.paths = dict()

        for cl in classes: 
            current_paths = sorted(glob.glob(os.path.join(path_to_dataset,cl,"*.jpg")))
            self.paths[cl] = current_paths
            
    def getImagePath(self,cl,idx):
        return self.paths[cl][idx]
    
    def getClasses(self):
        return sorted(self.paths.keys())
        
    def showImage(self,class_name,image_number):
        im = sio.imread(self.getImagePath(class_name,image_number))
        plt.figure()
        plt.imshow(im)
        plt.show()
    
    def getNumberOfClasses(self):
        return len(self.paths)
    
    def getClassLength(self,cl):
        return len(self.paths[cl])
    
    def getLength(self):
        return sum([len(x) for x in self.paths.values()])
    
    def restrictToClasses(self,classes):
        new_paths = {cl:self.paths[cl] for cl in classes}
        self.paths = new_paths
    
    def splitTrainingTest(self,percent_train):
        training_paths = dict() 
        test_paths = dict() 
        for cl in self.getClasses(): 
            paths = self.paths[cl] 
            shuffled_paths = np.random.permutation(paths)
            split_idx = int(len(shuffled_paths)*percent_train) 
            training_paths[cl] = shuffled_paths[0:split_idx]
            test_paths[cl] = shuffled_paths[split_idx::]
        
        training_dataset = copy(self)
        training_dataset.paths = training_paths
        
        test_dataset = copy(self)
        test_dataset.paths = test_paths
        return training_dataset,test_dataset