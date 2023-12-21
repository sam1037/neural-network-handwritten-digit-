import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

print("hi")

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data() #x is the image, y is the digit

print(x_train.shape, y_train.shape)

#normailize the image
#x_train = tf.keras.utils.normalize(x_train, axis = 1)
#x_test = tf.keras.utils.normalize(x_test, axis = 1)

x_train = np.where(x_train>=100, 1, 0)
x_test = np.where(x_test>=100, 1, 0)

for x in x_train[0:1]:
    print(x, type(x), x.shape)   


#build and train the model
"""
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape = (28,28))) #add the input layer
model.add(tf.keras.layers.Dense(20, activation="relu")) #hidden layer
model.add(tf.keras.layers.Dense(128, activation="relu")) #hidden layer
model.add(tf.keras.layers.Dense(128, activation="relu")) #hidden layer
model.add(tf.keras.layers.Dense(10, activation="softmax")) #output layer"""

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = (28,28)),
    tf.keras.layers.Dense(20, activation="relu"),
    tf.keras.layers.Dense(20, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics= ["accuracy"])

model.fit(x_train, y_train, epochs = 3)

model.save("test model for handwritten")

loss, accuracy = model.evaluate(x_test, y_test)
print(f"loss: {loss}, accuracy: {accuracy}")

prediction = model.predict(x_test)
print(x_test.shape)
print(prediction[0], type(prediction))
