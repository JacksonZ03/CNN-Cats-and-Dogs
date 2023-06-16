from keras import Sequential
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from keras.optimizers import *
from keras.models import *
import keras
import numpy as np
import os

img_size = 200 # number of pixels for width and height

#Random Seed
np.random.seed(17897)


training_path = os.getcwd() + "/cats and dogs images/train"
testing_path = os.getcwd() + "/cats and dogs images/test"

#Loads the Model
model = load_model('trained_model.h5')


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

#Recompiles model
#model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])


#Checkpoint
filepath = os.getcwd() + "/trained_model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min', save_freq="epoch")

#Fitting the model to the dataset (Training the Model)
model.fit(x = training_dataset, steps_per_epoch = 2200, validation_data=testing_dataset, validation_steps=100, epochs = 50, callbacks=[checkpoint], verbose = 1)


# evaluate model on training dataset
_,acc = model.evaluate(training_dataset, steps=len(training_dataset), verbose=1)
print("Accuracy on training dataset:")
print('> %.3f' % float(acc * 100.0))


#evaluate model on testing dataset
_,acc = model.evaluate(testing_dataset, steps=len(testing_dataset), verbose=1)
print("Accuracy on testing dataset:")
print('> %.3f' % (acc * 100.0))





