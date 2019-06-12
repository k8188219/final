import numpy as np
import scipy.misc
import csv
import cv2
import io
import glob

class DataLoader():
    def __init__(self, img_res=96):
        self.img_res = img_res
        
        self.train_filesT = []
        self.coordsT = []
        self.train_filesV = []
        self.coordsV = []
        

    def load_data(self, batch_size=1):
        self.loadAllCSV()
        if(len(self.train_filesV) == 0):
          return [],[]
        batch = self.train_filesV
        labels = self.coordsV
            
        imgs = []
        for img_path in batch:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224))
            imgs.append(img)
        imgs = np.array(imgs) / 127.5 - 1.
        
        Xtr_label = labels
        Xtr = imgs
        
        return Xtr, self.train_filesV

    def one_hot_encode(self, y, num_classes=2):
        return np.squeeze(np.eye(num_classes)[y.reshape(-1)])

    def loadAllCSV(self):
        self.coordsV = np.array([self.one_hot_encode(np.array([1])[0], num_classes=15)])
        self.train_filesV = glob.glob("./images/*")

