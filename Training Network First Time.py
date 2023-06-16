from keras import Sequential
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from keras.optimizers import *
import keras
import numpy as np
import os

img_size = 200 # number of pixels for width and height

#Random Seed
np.random.seed(123456789)


training_path = os.getcwd() + "/cats and dogs images/train"
testing_path = os.getcwd() + "/cats and dogs images/test"

#Defines the Model
model = Sequential([
        Conv2D(filters=20, kernel_size=(3,3), activation="relu", padding="same", input_shape=(img_size,img_size,3)),
        MaxPool2D(pool_size=(2,2), strides=2),
        Conv2D(filters=40, kernel_size=(3,3), activation="relu", padding="same"),
        MaxPool2D(pool_size=(2,2), strides=2),
        Conv2D(filters=80, kernel_size=(3,3), activation="relu", padding="same"),
        MaxPool2D(pool_size=(2,2), strides=2),
        Conv2D(filters=160, kernel_size=(3,3), activation="relu", padding="same"),
        MaxPool2D(pool_size=(2,2), strides=2),
        Conv2D(filters=320, kernel_size=(3,3), activation="relu", padding="same"),
        MaxPool2D(pool_size=(2,2), strides=2),
        Conv2D(filters=640, kernel_size=(3,3), activation="relu", padding="same"),
        MaxPool2D(pool_size=(2,2), strides=2),
        Flatten(),
        Dense(640, activation="relu"),
        Dense(2, activation="softmax")
])


#Scales the pixel values to between 0 to 1
datagen = ImageDataGenerator(rescale=1.0/255.0)

Batch_size = 10

#Prepares Training Data
training_dataset = datagen.flow_from_directory(directory = training_path,
                                               target_size=(img_size,img_size),
                                               classes = ["cat","dog"],
                                               class_mode = "categorical",
                                               batch_size = Batch_size)

#Prepares Testing Data
testing_dataset = datagen.flow_from_directory(directory = testing_path,
                                              target_size=(img_size,img_size),
                                              classes = ["cat","dog"],
                                              class_mode = "categorical",
                                              batch_size = Batch_size)


#Compiles the model
#model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=['accuracy'])
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
#model.compile(loss="mse", optimizer="sgd", metrics=[keras.metrics.MeanSquaredError()])

#Checkpoint
filepath = os.getcwd() + "/trained_model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min', save_freq="epoch")

#Fitting the model to the dataset (Training the Model)
model.fit(x = training_dataset, steps_per_epoch = 2200,
          validation_data=testing_dataset, validation_steps=100,
          epochs = 30, callbacks=[checkpoint], verbose = 1)


# evaluate model on training dataset
_,acc = model.evaluate_generator(training_dataset, steps=len(training_dataset), verbose=0)
print("Accuracy on training dataset:")
print('> %.3f' % (acc * 100.0))


#evaluate model on testing dataset
_,acc = model.evaluate_generator(testing_dataset, steps=len(testing_dataset), verbose=0)
print("Accuracy on testing dataset:")
print('> %.3f' % (acc * 100.0))

input()