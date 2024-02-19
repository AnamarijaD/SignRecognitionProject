import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Input
from keras import Sequential
from keras.losses import MeanSquaredError, BinaryCrossentropy ,SparseCategoricalCrossentropy
from tensorflow import keras as k
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TEST_DATA_PATH = '/home/ana/Desktop/fx/siap/sign_language_detection/siap-project/data/sign_mnist_test.csv'
TRAIN_DATA_PATH = '/home/ana/Desktop/fx/siap/sign_language_detection/siap-project/data/sign_mnist_train.csv'
RANDOM_SEED = 42

training_data = pd.read_csv(TRAIN_DATA_PATH)

testing_data = pd.read_csv(TEST_DATA_PATH)

x_= training_data.drop(columns=['label'])
y_=training_data['label']

x_test= testing_data.iloc[:,1:]
y_test=testing_data.iloc[:,0]

model=Sequential([k.layers.Input(shape=(784,)),k.layers.BatchNormalization(),k.layers.Dropout(.3),k.layers.Dense(300,activation='elu',kernel_initializer='he_normal',),k.layers.BatchNormalization(),k.layers.Dropout(.3),k.layers.Dense(300,activation='elu',kernel_initializer='he_normal'),k.layers.BatchNormalization(),k.layers.Dropout(.3),k.layers.Dense(25,activation='softmax')])
model.compile(optimizer=k.optimizers.Adam(),loss=k.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['accuracy'])
model.summary()

k.backend.set_learning_phase(1)
hist=model.fit(x_,y_,epochs=1000,validation_split=.2, callbacks=[k.callbacks.EarlyStopping(patience=10,restore_best_weights=True)])

model.evaluate(x_test,y_test)

torch.save(model, 'models/ANN')