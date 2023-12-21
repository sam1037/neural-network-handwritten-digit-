import tkinter as tk
from PIL import ImageGrab
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL

root = tk.Tk()

#load the model
model = tf.keras.models.load_model("test model for handwritten")
#model = tf.keras.models.load_model('floder/mnist.h5')


#set the size
root.geometry("800x800")
#set title
root.title("some randome title")


#the canvas
canvas = tk.Canvas(root, width=600, height=600, highlightthickness=2, highlightbackground="black", bg="white")
canvas.pack()

canvas.bind("<B1-Motion>", lambda event: canvas.create_oval(event.x-15, event.y-15, event.x+15, event.y+15, fill='black'))


#clear the drawing
def clear_drawing():
    canvas.delete("all")

clear_button = tk.Button(root, text="Clear Drawing", command=clear_drawing)
clear_button.pack()



#save the drawing without any normalization
def save_drawing():
    x = root.winfo_rootx() + canvas.winfo_x()
    y = root.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()
    drawing = ImageGrab.grab((x, y, x1, y1))
    drawing.save("user_drawing.png")


save_button = tk.Button(root, text="Save Drawing", command=save_drawing)
save_button.pack()


#prediction (normalize on the go)
def predict_digit():
    #normalize the image
    image = cv2.imread("user_drawing.png")
    image = cv2.resize(image, (28,28))


    image = image[:,:,0]
    #print(image.shape, "SHAOPEEEE")
    
    threshold = 240
    image = np.invert(np.array([image]))
    image = np.where(image >= threshold, 1, 0)
    print(image, image.shape)


    #image = image/255.0
    #image = np.where(image >= threshold, 1, 0)
    #print(image, type(image))
    #print(image[0], type(image[0]))

    #image = image[0]

    #image processing

    #prediction
    prediction = model.predict(image)
    print(prediction.shape, prediction)
    prediction = prediction[0]
    print(prediction.shape, prediction, type(prediction))
    for i in prediction:
        print(i)

    print()
    print(prediction.max())
    print(np.sum(prediction))
    print(np.argmax(prediction))


    confidence_score = prediction.max()/ np.sum(prediction)
    print(f"The prediction for user drawing is {np.argmax(prediction)} with confidence {confidence_score}")
    plt.imshow(image[0], cmap = plt.cm.binary)
    plt.show()

predict_button = tk.Button(root, text = "Predict Digit", command=predict_digit)
predict_button.pack()



root.mainloop()