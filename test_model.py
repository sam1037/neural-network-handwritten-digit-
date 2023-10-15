import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#load the model
model = tf.keras.models.load_model("test model for handwritten")

print("hi")

#test with mnsit data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data() #x is the image, y is the digit
x_test = tf.keras.utils.normalize(x_test, axis = 1)
print(x_test[0])
plt.imshow(x_test[0], interpolation='nearest')
plt.show()

loss, accuracy = model.evaluate(x_test, y_test)
print(f"loss: {loss}, accuracy: {accuracy}")

#test the model with manually created test data
file_digit_dict = {"1_3": 3, "2_5": 5, "3_9": 9, "4_6": 6, "5_7": 7, "6_1": 1, "7_3": 3, "8_2": 2, "9_0": 0, "10_4": 4}

correct_number = 0;

for file_name in file_digit_dict:
    try:
        image = cv2.imread(f"manually created test data/{file_name}.png")[:,:,0]
        #print(image, type(image))
        image = np.invert(np.array([image]))
        #normalize the image
        print(image, type(image))
        prediction = model.predict(image)
        print(f"The prediction for {file_name}.png is {np.argmax(prediction)}")
        print(f"The correct answer is {file_digit_dict[file_name]}")
        if np.argmax(prediction) == file_digit_dict[file_name]:
            correct_number +=1
        #plt.imshow(image[0], cmap = plt.cm.binary)
        #plt.show()
    except:
        print("Error!")


print(f"{correct_number} correct predictions are made")