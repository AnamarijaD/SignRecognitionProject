import pandas as pd 
import numpy as np 
# import plotly.express as px
import matplotlib.pylab as plt
from pandas.api.types import CategoricalDtype

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Subset, TensorDataset
from utils import *
import keras
from sklearn.model_selection import train_test_split
import tensorflow as tf 

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TEST_DATA_PATH = '/home/ana/Desktop/fx/siap/sign_language_detection/siap-project/data/sign_mnist_test.csv'
TRAIN_DATA_PATH = '/home/ana/Desktop/fx/siap/sign_language_detection/siap-project/data/sign_mnist_train.csv'
RANDOM_SEED = 42

training_data = pd.read_csv(TRAIN_DATA_PATH)

testing_data = pd.read_csv(TEST_DATA_PATH)


# take labels out of the date 
train_x = training_data.drop("label",axis=1)
train_y= training_data["label"]
test_x = testing_data.drop("label",axis=1)
test_y= testing_data["label"]
train_x.head()

# splitting the test into dev/blind sets
dev_x,blind_x ,dev_y, blind_y = train_test_split(test_x,test_y,test_size=.8,stratify=test_y)


# afinal step.. save the labels before one-hot-encoding 
test_classes= blind_y
dev_clasees = dev_y

# frist conver the date frame into numpy array
train_x = train_x.to_numpy()
dev_x   = dev_x.to_numpy()
blind_x    = blind_x.to_numpy()

# second : reshape the array
train_x = train_x.reshape(-1,28,28,1)
dev_x   = dev_x.reshape(-1,28,28,1)
blind_x    = blind_x.reshape(-1,28,28,1)

labels = training_data["label"].value_counts().sort_index(ascending=True)

def identity_block(X, f, filters, training=True):
  # filter of the three convs 
  f1,f2,f3 = filters
  X_shortcut = X 
  
  # first component 
  X = tf.keras.layers.Conv2D(filters = f1, kernel_size = 1, strides = (1,1), padding = 'valid')(X)
  X = tf.keras.layers.BatchNormalization(axis = 3)(X, training = training) # Default axis
  X = tf.keras.layers.Activation('relu')(X)
  # second component 
  X = tf.keras.layers.Conv2D(filters = f2, kernel_size = f, strides = (1,1), padding = 'same')(X)
  X = tf.keras.layers.BatchNormalization(axis = 3)(X, training = training) # Default axis
  X = tf.keras.layers.Activation('relu')(X)
  # third component 
  X = tf.keras.layers.Conv2D(filters = f3, kernel_size = 1, strides = (1,1), padding = 'valid')(X)
  X = tf.keras.layers.BatchNormalization(axis = 3)(X, training = training) # Default axis
  # adding the two paths 
  X = tf.keras.layers.Add()([X_shortcut,X])
  X = tf.keras.layers.Activation('relu')(X)
  # return the las tensor
  return X

def convolutional_block(X, f, filters, s=2,training=True):
  # filter of the three convs 
  f1,f2,f3 = filters
  X_shortcut = X 
  
  # first component 
  X = tf.keras.layers.Conv2D(filters = f1, kernel_size = 1, strides = (1,1), padding = 'valid')(X)
  X = tf.keras.layers.BatchNormalization(axis = 3)(X, training = training) # Default axis
  X = tf.keras.layers.Activation('relu')(X)
  # second component 
  X = tf.keras.layers.Conv2D(filters = f2, kernel_size = f, strides = (s,s), padding = 'same')(X)
  X = tf.keras.layers.BatchNormalization(axis = 3)(X, training = training) # Default axis
  X = tf.keras.layers.Activation('relu')(X)
  # third component 
  X = tf.keras.layers.Conv2D(filters = f3, kernel_size = 1, strides = (1,1), padding = 'valid')(X)
  X = tf.keras.layers.BatchNormalization(axis = 3)(X, training = training) # Default axis
  # converting the input volume to the match the last output for adding
  X_shortcut =tf.keras.layers.Conv2D(filters = f3, kernel_size = 1, strides = (s,s), padding = 'valid')(X_shortcut)
  X_shortcut = tf.keras.layers.BatchNormalization(axis = 3)(X_shortcut, training = training)
  X = tf.keras.layers.Add()([X_shortcut,X])
  X = tf.keras.layers.Activation('relu')(X)
  # last , add the two tensors
  X = tf.keras.layers.Add()([X, X_shortcut])
  X = tf.keras.layers.Activation('relu')(X)
  
  # return the las tensor
  return X

# UNQ_C3
# GRADED FUNCTION: ResNet50

def ResNet50(input_shape = (28, 28, 1), classes =len(labels) ):
    
  # Define the input as a tensor with shape input_shape
  X_input = tf.keras.Input(input_shape)
  
  # Zero-Padding
  X = tf.keras.layers.ZeroPadding2D((3, 3))(X_input)
  
  # Stage 1
  X = tf.keras.layers.Conv2D(64, (5, 5), strides = (1, 1))(X)
  X = tf.keras.layers.BatchNormalization(axis = 3)(X)
  X = tf.keras.layers.Activation('relu')(X)
  X = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(X)

  # Stage 2
  X = convolutional_block(X, f = 3, filters = [64, 64, 256], s = 1)
  # note: in building the identity block we make sure the input volume has the same dimesions as the output volume
  # means , the last filter of the block must be the same as the last filter of the previous block
  X = identity_block(X, 3, [64, 64, 256])
  X = identity_block(X, 3, [64, 64, 256])

  
  ## Stage 3 
  X = convolutional_block(X, f = 3, filters = [128,128,512], s = 2)
  X = identity_block(X, 3, [128,128,512]) 
  X = identity_block(X, 3, [128,128,512])
  X = identity_block(X, 3, [128,128,512]) 
  
  ## Stage 4
  X = convolutional_block(X, f = 3, filters =  [256, 256, 1024], s = 2) 
  X = identity_block(X, 3,  [256, 256, 1024])  
  X = identity_block(X, 3,  [256, 256, 1024])  
  X = identity_block(X, 3,  [256, 256, 1024])  
  X = identity_block(X, 3,  [256, 256, 1024])  
  X = identity_block(X, 3,  [256, 256, 1024])  

  ## Stage 5 
#     X = convolutional_block(X, f = 3, filters =   [512, 512, 2048], s = 1)  
#     X = identity_block(X, 3,   [512, 512, 2048]) 
#     X = identity_block(X, 3,   [512, 512, 2048])  
  
  ## addd average pool layer
  X = tf.keras.layers.AveragePooling2D((2,2))(X)

  # output layer
  X = tf.keras.layers.Flatten()(X)
  X = tf.keras.layers.Dense(classes, activation='softmax')(X)
  
  
  # Create model
  model = tf.keras.Model(inputs = X_input, outputs = X)

  return model

model =ResNet50()
print(model.summary())

# onehot encoding the labels
# for that lsets use pandas.getdummies method
train_y = pd.pandas.get_dummies(train_y)
dev_y   = pd.pandas.get_dummies(dev_y)
blind_y = pd.pandas.get_dummies(blind_y)

train_y.columns 

dev_y.columns 

# now compiling the model
model.compile(optimizer="adam",metrics=["accuracy"],loss = "categorical_crossentropy")

# data augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,  
        zoom_range = 0.1,  
        width_shift_range=0.1, 
        height_shift_range=0.1)

best_model =tf.keras.callbacks.ModelCheckpoint("best.h5",monitor="val_accuracy")

history = model.fit_generator(datagen.flow(train_x,train_y,batch_size=64),validation_data =datagen.flow(dev_x,dev_y),epochs =10,callbacks=[best_model])

model.load_weights("best.h5")

X = range(len(history.history["loss"]))
train_loss = history.history["loss"]
val_loss =history.history["val_loss"]
train_acc = history.history["accuracy"]
val_accuracy =history.history["val_accuracy"]

model.evaluate(blind_x,blind_y)

torch.save(model, 'models/RESNET')

