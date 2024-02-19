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

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TEST_DATA_PATH = '/home/ana/Desktop/fx/siap/sign_language_detection/siap-project/data/sign_mnist_test.csv'
TRAIN_DATA_PATH = '/home/ana/Desktop/fx/siap/sign_language_detection/siap-project/data/sign_mnist_train.csv'
RANDOM_SEED = 42

training_data = pd.read_csv(TRAIN_DATA_PATH)

testing_data = pd.read_csv(TEST_DATA_PATH)

class_labels = np.unique(training_data["label"].values)

num_of_objects = np.bincount(training_data["label"].values)

training_data["label"] = training_data["label"].apply(adjust_class_labels)
np.unique(training_data["label"].values)
np.bincount(training_data["label"].values)

testing_data["label"] = testing_data["label"].apply(adjust_class_labels)
np.unique(testing_data["label"].values)
np.bincount(testing_data["label"].values)

target = training_data["label"].values
features = training_data.drop("label", axis=1).values

target_test = testing_data["label"].values
features_test = testing_data.drop("label", axis=1).values

features = features.reshape(-1, 1, 28, 28)
features_test = features_test.reshape(-1, 1, 28, 28)

features_scaled = features / 255
features_test_scaled = features_test / 255

y_train = torch.from_numpy(target).float()
x_train = torch.from_numpy(features_scaled).float()

y_test = torch.from_numpy(target_test).float()
x_test = torch.from_numpy(features_test_scaled).float()

sign_lang_mnist_dataset = TensorDataset(x_train, y_train)

testing_dataset = TensorDataset(x_test, y_test)

training_dataset = Subset(
  sign_lang_mnist_dataset,
  torch.arange(10000, len(sign_lang_mnist_dataset)),
)

validation_dataset = Subset(sign_lang_mnist_dataset, torch.arange(10000))

'''Visualizing the data'''
# fig = plt.figure(figsize=(15, 6))
# for i, (data, label) in itertools.islice(enumerate(training_dataset), 10):
#     ax = fig.add_subplot(2, 5, i + 1)
#     ax.set_xticks([]); ax.set_yticks([])
#     ax.imshow(data.numpy().reshape(28, 28), cmap='gray_validation_dataset = Subset(sign_lang_mnist_dataset, torch.arange(10000))r')
#     ax.set_title(f'True label = {int(label)}', size=15)
# plt.suptitle("Training dataset examples", fontsize=20)
# plt.tight_layout()
# plt.show()

'''Plotting validation examples'''
# fig = plt.figure(figsize=(15, 6))
# for i, (data, label) in itertools.islice(enumerate(validation_dataset), 10):
#     ax = fig.add_subplot(2, 5, i + 1)
#     ax.set_xticks([]); ax.set_yticks([])
#     ax.imshow(data.numpy().reshape(28, 28), cmap='gray_r')
#     ax.set_title(f'True label = {int(label)}', size=15)
# plt.suptitle("Validation dataset examples", fontsize=20)
# plt.tight_layout()
# plt.show()

'''Plotting testing examples'''
# fig = plt.figure(figsize=(15, 6))
# for i, (data, label) in itertools.islice(enumerate(testing_dataset), 10):
#     ax = fig.add_subplot(2, 5, i + 1)
#     ax.set_xticks([]); ax.set_yticks([])
#     ax.imshow(data.numpy().reshape(28, 28), cmap='gray_r')
#     ax.set_title(f'True label = {int(label)}', size=15)
# plt.suptitle("Testing dataset examples", fontsize=20)
# plt.tight_layout()
# plt.show()

'''Plotting class labels shares(training data)'''
# training_labels = training_dataset[:][1].int()
# unique_training_labels = np.unique(training_labels)
# training_labels_count = np.bincount(training_labels)
# n_train = training_labels.shape[0]
# labels_info_share = pd.Series(
#     training_labels_count, index=unique_training_labels
# ) / n_train
# labels_info_share.plot(kind="bar")
# plt.xticks(rotation=0)
# plt.title("Shares of classes (training data)", fontsize=15)
# plt.xlabel("Class label (Label for the sign)")
# plt.ylabel("Proportion of objects")
# plt.tight_layout()
# plt.show()

'''Plotting class labels shares(validation data)'''
# validation_labels = validation_dataset[:][1].int()
# unique_validation_labels = np.unique(validation_labels)
# validation_labels_count = np.bincount(validation_labels)
# n_valid = validation_labels.shape[0]
# labels_info_share = pd.Series(
#     validation_labels_count, index=unique_validation_labels
# ) / n_valid
# labels_info_share.plot(kind="bar", color="orange")
# plt.xticks(rotation=0)
# plt.title("Shares of classes (validation data)", fontsize=15)
# plt.xlabel("Class label (Label for the sign)")
# plt.ylabel("Proportion of objects")
# plt.tight_layout()
# plt.show()

'''Plotting class labels shares(testing data)'''
# testing_labels = testing_dataset[:][1].int()
# unique_testing_labels = np.unique(testing_labels)
# testing_labels_count = np.bincount(testing_labels)
# n_test = testing_labels.shape[0]
# labels_info_share = pd.Series(
#     testing_labels_count, index=unique_testing_labels
# ) / n_test
# labels_info_share.plot(kind="bar", color="green")
# plt.xticks(rotation=0)
# plt.title("Shares of classes (testing data)", fontsize=15)
# plt.xlabel("Class label (Label for the sign)")
# plt.ylabel("Proportion of objects")
# plt.tight_layout()
# plt.show()

torch.manual_seed(RANDOM_SEED)
batch_size = 64

training_dataloader = DataLoader(
    training_dataset,
    batch_size=batch_size, 
    shuffle=True,
)

validation_dataloader = DataLoader(
    validation_dataset,
    batch_size=batch_size, 
    shuffle=False,
)

testing_dataloader = DataLoader(
    testing_dataset,
    batch_size=batch_size, 
    shuffle=False,
)

sign_mnist_classifier = nn.Sequential()

# Convolutional Layer #1
sign_mnist_classifier.add_module(
  "conv1",
  nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
)

# Activation for Convolutional Layer #1
sign_mnist_classifier.add_module("relu1", nn.ReLU())

# Max-Pooling Layer #1
sign_mnist_classifier.add_module("pool1", nn.MaxPool2d(kernel_size=2))

# Convolutional Layer #2
sign_mnist_classifier.add_module(
  "conv2", 
  nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
)

# Activation for Convolutional Layer #2
sign_mnist_classifier.add_module("relu2", nn.ReLU())

# Max-Pooling Layer #2
sign_mnist_classifier.add_module("pool2", nn.MaxPool2d(kernel_size=2))

# Flatten Layer
sign_mnist_classifier.add_module("flatten", nn.Flatten())

# Fully Connected Layer #1
sign_mnist_classifier.add_module('fc1', nn.Linear(3136, 1024))

# Activation for Fully Connected Layer #1
sign_mnist_classifier.add_module('relu3', nn.ReLU())

# Dropout
sign_mnist_classifier.add_module('dropout', nn.Dropout(p=0.5))

# Fully Connected Layer #2
sign_mnist_classifier.add_module('fc2', nn.Linear(1024, 24))

# print(sign_mnist_classifier.eval())    

def train(
  model, 
  loss_func, 
  optimizer,
  training_dataloader,
  validation_dataloader,
  epochs=10,
  enable_logging=False,
  device=DEVICE,
  ):
  
  # Preallocating the arrays for losses and accuracies
      loss_history_train = [0] * epochs
      accuracy_history_train = [0] * epochs
      loss_history_valid = [0] * epochs
      accuracy_history_valid = [0] * epochs
      
      # Launching the algorithm
      for epoch in range(epochs):
          
          # Enabling training mode
          model.train()
          
          # Considering each batch for the current epoch
          for x_batch, y_batch in training_dataloader:
              
              # Moving data to GPU
              x_batch = x_batch.to(device)
              y_batch = y_batch.to(device)
              
              # Generating predictions for the batch
              model_predictions = model(x_batch)
              
              # Computing the loss
              loss = loss_func(model_predictions, y_batch.long())
              
              # Computing gradients
              loss.backward()
              
              # Updating parameters using gradients
              optimizer.step()
              
              # Resetting the gradients to zero
              optimizer.zero_grad()
              
              # Adding the batch-level loss and accuracy to history
              loss_history_train[epoch] += loss.item() * y_batch.size(0)
              is_correct = (
                  torch.argmax(model_predictions, dim=1) == y_batch
              ).float()
              accuracy_history_train[epoch] += is_correct.sum().cpu()
              
              # Computing epoch-level loss and accuracy
          loss_history_train[epoch] /= len(training_dataloader.dataset)
          accuracy_history_train[epoch] /= len(training_dataloader.dataset)
          
          # Enabling evaluation mode
          model.eval()
          
          # Testing the CNN on the validation set
          with torch.no_grad():
              
              # Considering each batch for the current epoch
              for x_batch, y_batch in validation_dataloader:
                  
                  # Moving data to GPU
                  x_batch = x_batch.to(device)
                  y_batch = y_batch.to(device)
                  
                  # Generating predictions for the batch
                  model_predictions = model(x_batch)
              
                  # Computing the loss
                  loss = loss_func(model_predictions, y_batch.long())
                  
                  # Adding the batch-level loss and accuracy to history
                  loss_history_valid[epoch] += loss.item() * y_batch.size(0)
                  is_correct = (
                      torch.argmax(model_predictions, dim=1) == y_batch
                  ).float()
                  accuracy_history_valid[epoch] += is_correct.sum().cpu()
                  
          # Computing epoch-level loss and accuracy
          loss_history_valid[epoch] /= len(validation_dataloader.dataset)
          accuracy_history_valid[epoch] /= len(validation_dataloader.dataset)
          
          # Logging the training process
          if enable_logging:
              print(
                  "Epoch {}/{}\n"
                  "train_loss = {:.4f}, train_accuracy = {:.4f} | "
                  "valid_loss = {:.4f}, valid_accuracy = {:.4f}".format(
                  epoch + 1,
                  epochs,
                  loss_history_train[epoch], 
                  accuracy_history_train[epoch],
                  loss_history_valid[epoch],
                  accuracy_history_valid[epoch],
                  )
              )
              
      return (
          model, 
          loss_history_train, 
          accuracy_history_train,
          loss_history_valid,
          accuracy_history_valid,
      )
  
loss_func = nn.CrossEntropyLoss()

learning_rate = 0.001

optimizer = torch.optim.Adam(
  sign_mnist_classifier.parameters(), lr=learning_rate
)

torch.manual_seed(RANDOM_SEED)

(
  sign_mnist_classifier,
  loss_history_train, 
  accuracy_history_train,
  loss_history_valid,
  accuracy_history_valid 
  ) = train(
    model=sign_mnist_classifier,
    loss_func=loss_func,
    optimizer=optimizer,
    training_dataloader=training_dataloader,
    validation_dataloader=validation_dataloader,
    epochs=8,
    enable_logging=True,
)

torch.save(sign_mnist_classifier , "models/SignMNIST_Classification_Model_CNN")