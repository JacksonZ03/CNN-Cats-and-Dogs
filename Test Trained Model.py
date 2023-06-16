from keras.models import Sequential
from keras_preprocessing.image import *
from keras.layers import *
import tensorflow as tf
import numpy as np
from keras.layers.experimental.preprocessing import Rescaling
import os
import cv2
from keras.models import *

img_size = 200

#Load weights into new model
filepath = os.getcwd() + "/trained_model - 87% accuracy - 200 px.h5"

model = load_model(filepath)
print("Loaded model from disk")

#Scales the pixel values to between 0 to 1
#datagen = ImageDataGenerator(rescale=1.0/255.0)

#Prepares Testing Data

image_name = input("Please Type in the image name: ")

testing_dataset = cv2.imread(os.getcwd() + "/Place Image in Here To Test/"+ image_name)
#img = datagen.flow_from_directory(testing_dataset, target_size=(img_size,img_size))

img = cv2.resize(testing_dataset, (img_size,img_size))
newimg = np.asarray(img)
pixels = newimg.astype('float32')
pixels /= 255.0
print(pixels.shape)
pixels = np.expand_dims(pixels, axis=0)
print(pixels.shape)
prediction = model.predict(pixels)# or model(pixels) will also give a prediction
print(prediction)
output_class = np.argmax(prediction,axis = 1)

if (output_class == [1]):
    print("The image is a dog!")
elif(output_class == [0]):
    print("The image is a cat!")

input()
