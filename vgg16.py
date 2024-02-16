import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from keras.utils import to_categorical 
import keras as K
from keras.layers import Dense, Dropout,Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers import Dropout
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array, array_to_img

import torch
import torch.nn as nn
import torch.optim as optim
# from sklearn.metrics import classification_report
# from torch.utils.data import DataLoader, Subset, TensorDataset
# from utils import *
# import keras
# # from keras.applications import VGG16
# from keras.applications.vgg16 import preprocess_input, VGG16
# from torchvision import models, transforms

# DATA_DIR = r"C:\Users\Ana_Marija\Downloads\siap"
# TRAIN_DATA_DIR = os.path.join(DATA_DIR, "sign_mnist_train")
# TEST_DATA_DIR = os.path.join(DATA_DIR, "sign_mnist_test")
# TRAIN_DATA_PATH = os.path.join(TRAIN_DATA_DIR, "sign_mnist_train.csv")
# TEST_DATA_PATH = os.path.join(TEST_DATA_DIR, "sign_mnist_test.csv")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TEST_DATA_PATH = '/home/ana/Desktop/fx/siap/sign_language_detection/siap-project/data/sign_mnist_test.csv'
TRAIN_DATA_PATH = '/home/ana/Desktop/fx/siap/sign_language_detection/siap-project/data/sign_mnist_train.csv'
RANDOM_SEED = 42

training_data = pd.read_csv(TRAIN_DATA_PATH)

training_data.shape

testing_data = pd.read_csv(TEST_DATA_PATH)

x_train= training_data.values[0:,1:]
y_train = training_data.values[0:,0]

y_train = to_categorical(y_train)
x_train = x_train.reshape(-1,28,28,1)

x_train.shape

x_train = x_train/255

x_train = np.stack([x_train.reshape(x_train.shape[0],28,28)]*3, axis=3).reshape(x_train.shape[0],28,28,3)

x_train.shape

model = Sequential()

model.add(VGG16(weights='imagenet',
                  include_top=False, pooling = 'avg',  
                  input_shape=(48, 48, 3)
                 ))

model.add(Dense(25, activation = 'softmax'))

model.layers[0].trainable = False

model.compile(loss = K.losses.categorical_crossentropy, optimizer=K.optimizers.Adam(),
              metrics=['accuracy'])

x_train_tt = np.asarray([img_to_array(array_to_img(im, scale=True).resize((48,48))) for im in x_train])/225

x_train_tt.shape

model.fit(x_train_tt,y_train,epochs=5)

torch.save(model, 'VGG16')

x_test = testing_data.iloc[:,1:]
y_test = testing_data.iloc[:,0]

x_test = x_test.to_numpy().reshape(-1,28,28,1)
x_test = x_test/255

y_test = to_categorical(y_test)

x_test = np.stack([x_test.reshape(x_test.shape[0],28,28)]*3, axis=3).reshape(x_test.shape[0],28,28,3)

x_test_tt = np.asarray([img_to_array(array_to_img(im, scale=True).resize((48,48))) for im in x_test])/225

model.evaluate(x_test_tt,y_test)


