import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# Load data
TRAIN_DATA_PATH = r'C:\Users\Ana_Marija\Downloads\siap\SignRecognitionProject\data\sign_mnist_train.csv'
TEST_DATA_PATH = r'C:\Users\Ana_Marija\Downloads\siap\SignRecognitionProject\data\sign_mnist_test.csv'

training_data = pd.read_csv(TRAIN_DATA_PATH)
testing_data = pd.read_csv(TEST_DATA_PATH)

# Extract features and labels
train_x = training_data.drop("label", axis=1).to_numpy().reshape(-1, 28, 28, 1)
train_y = training_data["label"].to_numpy()
test_x = testing_data.drop("label", axis=1).to_numpy().reshape(-1, 28, 28, 1)
test_y = testing_data["label"].to_numpy()

# Preprocess images
def preprocess_images(images):
    images_resized = tf.image.resize(images, (32, 32))
    images_resized = tf.image.grayscale_to_rgb(images_resized)
    return images_resized

train_x_resized = preprocess_images(train_x)
test_x_resized = preprocess_images(test_x)

# One-hot encode the labels
label_binarizer = LabelBinarizer()
train_y_one_hot = label_binarizer.fit_transform(train_y)
test_y_one_hot = label_binarizer.transform(test_y)

# Data augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,  
        zoom_range=0.1,  
        width_shift_range=0.1, 
        height_shift_range=0.1)

# Load pre-trained ResNet50 without the classification layer
pretrained_resnet = tf.keras.applications.ResNet50(input_shape=(32, 32, 3), include_top=False, weights='imagenet')

# Freeze most of the layers of the pre-trained ResNet50
for layer in pretrained_resnet.layers[:-4]:
    layer.trainable = False

# Add your own classification layers
flatten_layer = tf.keras.layers.Flatten()(pretrained_resnet.output)
dense_layer = tf.keras.layers.Dense(256, activation='relu')(flatten_layer)
output_layer = tf.keras.layers.Dense(len(label_binarizer.classes_), activation='softmax')(dense_layer)

# Create a new model
custom_model = tf.keras.models.Model(inputs=pretrained_resnet.input, outputs=output_layer)

# Compile the model
custom_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callback to save the best model
best_model = tf.keras.callbacks.ModelCheckpoint("best.h5", monitor="val_accuracy", save_best_only=True)

# Train the model
history_custom = custom_model.fit(datagen.flow(train_x_resized, train_y_one_hot, batch_size=64),
                                  validation_data=datagen.flow(test_x_resized, test_y_one_hot),
                                  epochs=10,
                                  callbacks=[best_model])

# Evaluate the model on the test set
test_loss, test_accuracy = custom_model.evaluate(test_x_resized, test_y_one_hot)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

custom_model.save('models/RESNET-transferLearning')