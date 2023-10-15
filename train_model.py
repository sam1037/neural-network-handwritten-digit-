import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

print("hi")

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data() #x is the image, y is the digit

#normailize the image
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

#build and train the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape = (28,28))) #add the input layer
model.add(tf.keras.layers.Dense(128, activation="relu")) #hidden layer
model.add(tf.keras.layers.Dense(128, activation="relu")) #hidden layer
model.add(tf.keras.layers.Dense(10, activation="softmax")) #output layer

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics= ["accuracy"])

model.fit(x_train, y_train, epochs = 3)

model.save("test model for handwritten")
