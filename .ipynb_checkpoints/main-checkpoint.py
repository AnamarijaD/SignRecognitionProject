import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Subset, TensorDataset
from utils import *

DATA_DIR = r"C:\Users\Ana_Marija\Downloads\siap"
TRAIN_DATA_DIR = os.path.join(DATA_DIR, "sign_mnist_train")
TEST_DATA_DIR = os.path.join(DATA_DIR, "sign_mnist_test")
TRAIN_DATA_PATH = os.path.join(TRAIN_DATA_DIR, "sign_mnist_train.csv")
TEST_DATA_PATH = os.path.join(TEST_DATA_DIR, "sign_mnist_test.csv")
RANDOM_SEED = 42
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    # print("Using device: {}".format(DEVICE))

    # Loading the data
    training_data = pd.read_csv(TRAIN_DATA_PATH)
    # print(training_data.head())
    # print(training_data.shape)

    testing_data = pd.read_csv(TEST_DATA_PATH)
    # print(testing_data.head())
    # print(testing_data.shape)

    '''It can be seen from the unique class labels as well as the object number 
    counts that we do not have examples for class label 9, since such a sign can 
    only be captured by gestures. Besides, label 25 is also absent due to the same reason.'''
    # Displaying class labels (training data)
    class_labels = np.unique(training_data["label"].values)
    # Computing the number of objects for each class (training data)
    num_of_objects = np.bincount(training_data["label"].values)
    # print(num_of_objects)

    # Displaying class labels (testing data)
    class_labels = np.unique(testing_data["label"].values)
    # Computing the number of objects for each class (testing data)
    num_of_objects = np.bincount(testing_data["label"].values)
    # print(num_of_objects)

    # Adjusting the class labels (training data)
    training_data["label"] = training_data["label"].apply(adjust_class_labels)
    # Verifying the new class labels
    np.unique(training_data["label"].values)
    # Verifying the counts
    np.bincount(training_data["label"].values)

    # Adjusting the class labels (testing data)
    testing_data["label"] = testing_data["label"].apply(adjust_class_labels)
    np.unique(testing_data["label"].values)
    np.bincount(testing_data["label"].values)

    '''Creating datasets'''
    # Separating features from target (training)
    target = training_data["label"].values
    features = training_data.drop("label", axis=1).values

    # Separating features from target (testing)
    target_test = testing_data["label"].values
    features_test = testing_data.drop("label", axis=1).values

    # Reshaping the training/testing data for NN
    features = features.reshape(-1, 1, 28, 28)
    features_test = features_test.reshape(-1, 1, 28, 28)

    # Rescaling the features (training + testing)
    features_scaled = features / 255
    features_test_scaled = features_test / 255

    # Converting NumPy arrays to Torch tensors (training)
    y_train = torch.from_numpy(target).float()
    x_train = torch.from_numpy(features_scaled).float()

    # Converting NumPy arrays to Torch tensors (testing)
    y_test = torch.from_numpy(target_test).float()
    x_test = torch.from_numpy(features_test_scaled).float()

    # Initializing a TensorDataset (training)
    sign_lang_mnist_dataset = TensorDataset(x_train, y_train)

    # Initializing a TensorDataset (testing)
    testing_dataset = TensorDataset(x_test, y_test)

    # Splitting data into training/validation datasets
    training_dataset = Subset(
        sign_lang_mnist_dataset,
        torch.arange(10000, len(sign_lang_mnist_dataset)),
    )

    validation_dataset = Subset(sign_lang_mnist_dataset, torch.arange(10000))

    # Results are 17455, 10000, 7172
    # print(len(training_dataset)), print(len(validation_dataset)), print(len(testing_dataset))


    '''Visualizing the data'''
    # fig = plt.figure(figsize=(15, 6))
    # for i, (data, label) in itertools.islice(enumerate(training_dataset), 10):
    #     ax = fig.add_subplot(2, 5, i + 1)
    #     ax.set_xticks([]); ax.set_yticks([])
    #     ax.imshow(data.numpy().reshape(28, 28), cmap='gray_r')
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

    '''Creating dataloaders'''
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

    '''Building CNN'''
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

    print(sign_mnist_classifier.eval())

if __name__ == "__main__":
    main()
