from __future__ import print_function, division
from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
import datetime
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from data_loader import DataLoader
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.densenet import DenseNet121
from keras import backend as K
import cv2
import csv
import os
import numpy as np
import time
from keras.models import load_model
from keras import losses


class ConvolutionalNeuralNetworks():
    def __init__(self):
        # Input shape
        self.img_size = 224
        self.img_rows = self.img_size
        self.img_cols = self.img_size
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Configure data loader
        self.data_loader = DataLoader(img_res=self.img_size)
        # Build the network
        
        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
        
        self.CNN_Network = self.build_CNN_Network()
        self.CNN_Network.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
    def build_CNN_Network(self):

        def flatten(layer_input):
            d = Flatten()(layer_input)
            return d

        def dense(layer_input, f_size, dr=True, lastLayer=True):
            if lastLayer:
                d = Dense(f_size, activation='softmax')(layer_input)
            else:
                d = Dense(f_size, activation='linear')(layer_input)
                d = LeakyReLU(alpha=0.2)(d)
                if dr:
                    d = Dropout(0.5)(d)
            return d

        base_model = DenseNet121(include_top=False,input_shape=self.img_shape)
        
        d1 = base_model.output
     
        d2 = flatten(d1)
        d3 = dense(d2, f_size=120, dr=True, lastLayer=False)
        d4 = dense(d3, f_size=84, dr=True, lastLayer=False)
        d5 = dense(d4, f_size=15, dr=False, lastLayer=True)
        #m = ResNet50(include_top=True, weights=None, input_tensor=None, input_shape=None, pooling=None, classes=10)

        return Model(base_model.input, d5)
    def build_CNN_Network_VGG16(self):

        def flatten(layer_input):
            d = Flatten()(layer_input)
            return d

        def dense(layer_input, f_size, dr=True, lastLayer=True):
            if lastLayer:
                d = Dense(f_size, activation='softmax')(layer_input)
            else:
                d = Dense(f_size, activation='linear')(layer_input)
                d = LeakyReLU(alpha=0.2)(d)
                if dr:
                    d = Dropout(0.5)(d)
            return d

        base_model = ResNet50(include_top=False, input_shape=self.img_shape)
        d1 = base_model.output
        d2 = flatten(d1)
        d3 = dense(d2, f_size=120, dr=True, lastLayer=False)
        d4 = dense(d3, f_size=84, dr=True, lastLayer=False)
        d5 = dense(d4, f_size=15, dr=False, lastLayer=True)

        return Model(base_model.input, d5)
    
if __name__ == '__main__':
    print("start")
#     training model
#     my_CNN_Model = ConvolutionalNeuralNetworks()
#     my_CNN_Model.train(epochs=100, batch_size=20, sample_interval=50, skip=0)
    
    
#     testing model
    my_CNN = ConvolutionalNeuralNetworks()
    my_CNN.CNN_Network.load_weights('./cnn/CNN_Network_on_epoch_170.h5')
    while (True):
        time.sleep( 0.5 )
        imgs, labels = my_CNN.data_loader.load_data()
        if(len(imgs) == 0):
          continue
        pred_labels = my_CNN.CNN_Network.predict(imgs)
        print(labels, str(np.argmax(pred_labels, axis=1)))
        for i,file in enumerate(labels):
          fo = open('./result/'+file.split('/')[2], "w")
          s = str(np.argmax(pred_labels, axis=1)[i])
          print(s)
          fo.write( s )
          fo.close()
          os.remove(file)
          
    print("end")