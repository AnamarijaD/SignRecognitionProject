import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Subset, TensorDataset
from utils import *
import keras
# from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input, VGG16
from torchvision import models, transforms

TEST_DATA_PATH = '/home/ana/Desktop/fx/siap/sign_language_detection/siap-project/data/sign_mnist_test.csv'
RANDOM_SEED = 42

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

testing_data = pd.read_csv(TEST_DATA_PATH)
testing_data["label"] = testing_data["label"].apply(adjust_class_labels)
np.unique(testing_data["label"].values)
np.bincount(testing_data["label"].values)
target_test = testing_data["label"].values
features_test = testing_data.drop("label", axis=1).values
features_test = features_test.reshape(-1, 1, 28, 28)
features_test_scaled = features_test / 255
y_test = torch.from_numpy(target_test).float()
x_test = torch.from_numpy(features_test_scaled).float()

batch_size = 128
num_classes = 24
epochs = 50

testing_dataset = TensorDataset(x_test, y_test)

# validation_dataset = Subset(testing_dataset, torch.arange(10000))

testing_dataloader = DataLoader(
  testing_dataset,
  batch_size=batch_size, 
  shuffle=False,
)

def evaluate_test_by_batch(model, testing_dataloader, device=DEVICE):
    # Initializing the counter for accuracy
    accuracy_test = 0
    # Initializing a list for storing predictions
    test_predictions = []
    # Setting the model to the evaluation model
    model.eval()
    # Computing accuracy and predictions
    with torch.no_grad():
        for x_batch, y_batch in testing_dataloader:
            # Moving data to GPU
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            # Computing predictions for the batch
            test_batch_predictions = torch.argmax(model(x_batch), dim=1)
            # Adding the batch predictions to the prediction list
            test_predictions.append(test_batch_predictions)
            # Computing the test accuracy
            is_correct = (test_batch_predictions == y_batch).float()
            accuracy_test += is_correct.sum().cpu()
    
    # Transforming a list of tensors into one tensor
    test_predictions_tensor = torch.cat(test_predictions).cpu()
    # Finishing computing test accuracy
    accuracy_test /= len(testing_dataloader.dataset)
    
    return accuracy_test, test_predictions_tensor


sign_mnist_classifier = torch.load("SignMNIST_Classification_Model_CNN")

vgg16 = torch.load("VGG16")

accuracy_test, predictions_test = evaluate_test_by_batch(
    model=sign_mnist_classifier, testing_dataloader=testing_dataloader
)
print('*****cnn model*****')
print(f"Test accuracy: {accuracy_test:.4f}")
# print(f"Predictions test: {predictions_test:.4f}")
print('*****cnn model*****')

from keras.preprocessing.image import img_to_array, array_to_img

testing_data = pd.read_csv(TEST_DATA_PATH)

x_test = testing_data.iloc[:,1:]
y_test = testing_data.iloc[:,0]

x_test = x_test.to_numpy().reshape(-1,28,28,1)
x_test = x_test/255

y_test = keras.utils.to_categorical(y_test)

x_test = np.stack([x_test.reshape(x_test.shape[0],28,28)]*3, axis=3).reshape(x_test.shape[0],28,28,3)

x_test_tt = np.asarray([img_to_array(array_to_img(im, scale=True).resize((48,48))) for im in x_test])/225

vgg16.evaluate(x_test_tt,y_test)

