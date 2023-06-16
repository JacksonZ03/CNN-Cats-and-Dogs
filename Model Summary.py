from keras.models import *
import os


filepath = os.getcwd() + "/trained_model - 87% accuracy - 200 px.h5"
model = load_model(filepath)
summary = model.summary()
print(summary)
input()
