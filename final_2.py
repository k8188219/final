from __future__ import print_function, division
from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
import datetime
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
# from denseData_loader import DataLoader
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.densenet import DenseNet121
from keras import backend as K
import cv2
import numpy as np
from keras.models import load_model
import scipy.misc
from sklearn.metrics import mean_squared_error
from keras import losses
from sklearn.metrics import accuracy_score
import scipy.misc
import csv
import cv2
import io
from glob import glob

import numpy as np
import scipy.misc
import csv
import cv2
import io
from glob import glob


class DataLoader():
    def __init__(self, img_res=96):
        self.img_res = img_res

        self.train_filesT = []
        self.coordsT = []
        self.train_filesV = []
        self.coordsV = []
        self.loadAllCSV()

    def load_batch(self, batch_size=1):

        self.n_batches = int(len(self.coordsT) / batch_size)

        for i in range(self.n_batches - 1):
            path_u = self.train_filesT[i * batch_size:(i + 1) * batch_size]
            labels = self.coordsT[i * batch_size:(i + 1) * batch_size]

            imgs = []
            for img_path in path_u:
                img = cv2.imread('./train/' + img_path)
                img = cv2.resize(img, (224, 224))
                imgs.append(img)
            imgs = np.array(imgs) / 127.5 - 1.

            Xtr_label = labels
            Xtr = imgs

            yield Xtr, Xtr_label

    def load_data(self, batch_size=1):

        indices = (len(self.coordsV) * np.random.rand(batch_size)).astype(int)
        #       print(indices)

        self.train_filesV = np.array(self.train_filesV)

        images = self.train_filesV[indices, :]

        Xte = np.array(images) / 127.5 - 1.

        labels = []
        for label in self.coordsV[indices]:
            labels.append(label)

        Xte_label = np.array(labels)

        return Xte, Xte_label

    def one_hot_encode(self, y, num_classes=2):
        return np.squeeze(np.eye(num_classes)[y.reshape(-1)])

    def loadAllCSV(self):
        with open("./train.csv", "r") as csvfile:
            # 讀取 CSV 檔案內容
            train_data_dict = csv.reader(csvfile, delimiter=',')
            arr = []
            for index, row in enumerate(train_data_dict):
                path, label = row
                arr.append(int(label))
                self.train_filesT.append(path)

            for label in np.array(arr):
                self.coordsT.append(self.one_hot_encode(label, num_classes=15))
            self.coordsT = np.array(self.coordsT)

        with open("./train.csv", "r") as csvfile:
            # 讀取 CSV 檔案內容
            train_data_dict = csv.reader(csvfile, delimiter=',')
            arr = []
            for index, row in enumerate(train_data_dict):
                path, label = row
                arr.append(int(label))
                img = cv2.imread('./train/' + path)
                img = cv2.resize(img, (224, 224))
                self.train_filesV.append(img)

            for label in np.array(arr):
                self.coordsV.append(self.one_hot_encode(label, num_classes=15))
            self.coordsV = np.array(self.coordsV)


#         self.coordsV = np.array([self.one_hot_encode(np.array([14])[0], num_classes=15)])
#         self.train_filesV.append('11146295476254.jpg')

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

        base_model = DenseNet121(include_top=False, input_shape=self.img_shape)

        d1 = base_model.output

        d2 = flatten(d1)
        d3 = dense(d2, f_size=120, dr=True, lastLayer=False)
        d4 = dense(d3, f_size=84, dr=True, lastLayer=False)
        d5 = dense(d4, f_size=15, dr=False, lastLayer=True)
        # m = ResNet50(include_top=True, weights=None, input_tensor=None, input_shape=None, pooling=None, classes=10)

        return Model(base_model.input, d5)

    def train(self, epochs, batch_size=1, sample_interval=50, skip=0):
        start_time = datetime.datetime.now()

        for epoch in range(epochs):
            for batch_i, (imgs, coords) in enumerate(self.data_loader.load_batch(batch_size)):
                #  Training
                loss = self.CNN_Network.train_on_batch(imgs, coords)

                elapsed_time = datetime.datetime.now() - start_time
                print("[Epoch %d/%d] [Batch %d/%d] [Training loss: %f, Training acc: %3d%%] time: %s" % (
                    epoch + 1, epochs,
                    batch_i + 1, self.data_loader.n_batches - 1,
                    loss[0], 100 * loss[1],
                    elapsed_time))
                # If at save interval => do validation
                if (batch_i + 1) % sample_interval == 0:
                    self.validation(epoch, batch_i + 1)
            self.validation(epoch, batch_size)
            if (epochs < 50):
                continue
            else:
                self.CNN_Network.save_weights('./saved_model/CNN_Network_on_epoch_%d.h5' % (epoch))

    def validation(self, epoch, num_batch):
        Xte, Xte_labels = self.data_loader.load_data(batch_size=8)
        pred_labels = self.CNN_Network.predict(Xte)
        print("Validation acc: " + str(
            int(accuracy_score(np.argmax(Xte_labels, axis=1), np.argmax(pred_labels, axis=1)) * 100)) + "%")


if __name__ == '__main__':
    print("start")
    #    training model
    #     my_CNN_Model = ConvolutionalNeuralNetworks()
    #     my_CNN_Model.train(epochs=100, batch_size=16, sample_interval=50, skip=0)

    #     testing model
    my_CNN = ConvolutionalNeuralNetworks()
    my_CNN.CNN_Network.load_weights('./saved_model/CNN_Network_on_epoch_148.h5')
    imgs, labels = my_CNN.data_loader.load_data(batch_size=1024)
    pred_labels = my_CNN.CNN_Network.predict(imgs)
    print("Validation acc: " + str(
        int(accuracy_score(np.argmax(labels, axis=1), np.argmax(pred_labels, axis=1)) * 100)) + "%")

    print("end")
