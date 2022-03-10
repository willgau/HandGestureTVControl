import os

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd

# Sklearn
from sklearn.model_selection import train_test_split # Helps with organizing data for training
from sklearn.metrics import confusion_matrix # Helps present results as a confusion-matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten
# We need to get all the paths for the images to later load them
# imagepaths = []
#
# # Go through all the files and subdirectories inside a folder and save path to images inside list
# for root, dirs, files in os.walk(".", topdown=False):
#   for name in files:
#     path = os.path.join(root, name)
#     if path.endswith("png"): # We want only the images
#       imagepaths.append(path)
#
# print(len(imagepaths)) # If > 0, then a PNG image was loaded
# X = []  # Image data
# y = []  # Labels
#
# # Loops through imagepaths to load images and labels into arrays
# for path in imagepaths:
#   img = cv2.imread(path)  # Reads image and returns np.array
#   img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Converts into the corret colorspace (GRAY)
#   img = cv2.resize(img, (320, 120))  # Reduce image size so training can be faster
#   X.append(img)
#
#   # Processing label in image path
#   category = path.split("\\")[3]
#   label = int(category.split("_")[0][1])  # We need to convert 10_down to 00_down, or else it crashes
#   y.append(label)
#
# # Turn X and y into np.array to speed up train_test_split
# X = np.array(X, dtype="uint8")
# X = X.reshape(len(imagepaths), 120, 320, 1)  # Needed to reshape so CNN knows it's different images
# y = np.array(y)
#
# print("Images loaded: ", len(X))
# print("Labels loaded: ", len(y))
#
# print(y[0], imagepaths[0])  # Debugging
#
# ts = 0.3 # Percentage of images that we want to use for testing. The rest is used for training.
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=42)
#
# model = Sequential()
# model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(120, 320, 1)))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(10, activation='softmax'))
#
#
# # Configures the model for training
# model.compile(optimizer='RMSprop', # Optimization routine, which tells the computer how to adjust the parameter values to minimize the loss function.
#               loss='sparse_categorical_crossentropy', # Loss function, which tells us how bad our predictions are.
#               metrics=['accuracy']) # List of metrics to be evaluated by the model during training and testing.
#
#
#
# # Trains the model for a given number of epochs (iterations on a dataset) and validates it.
# model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=2, validation_data=(X_test, y_test))
#
# # Save entire model to a HDF5 file
# model.save('handrecognition_model.h5')
#
# predictions = model.predict(X_test) # Make predictions towards the test set
#
# np.argmax(predictions[0]), y_test[0] # If same, got it right
#
#
# # Function to plot images and labels for validation purposes
# def validate_9_images(predictions_array, true_label_array, img_array):
#   # Array for pretty printing and then figure size
#   class_names = ["down", "palm", "l", "fist", "fist_moved", "thumb", "index", "ok", "palm_moved", "c"]
#   plt.figure(figsize=(15, 5))
#
#   for i in range(1, 10):
#     # Just assigning variables
#     prediction = predictions_array[i]
#     true_label = true_label_array[i]
#     img = img_array[i]
#     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#
#     # Plot in a good way
#     plt.subplot(3, 3, i)
#     plt.grid(False)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(img, cmap=plt.cm.binary)
#
#     predicted_label = np.argmax(prediction)  # Get index of the predicted label from prediction
#
#     # Change color of title based on good prediction or not
#     if predicted_label == true_label:
#       color = 'blue'
#     else:
#       color = 'red'
#
#     plt.xlabel("Predicted: {} {:2.0f}% (True: {})".format(class_names[predicted_label],
#                                                           100 * np.max(prediction),
#                                                           class_names[true_label]),
#                color=color)
#   plt.show()
#
# validate_9_images(predictions, y_test, X_test)

class_names = ["down", "palm", "l", "fist", "fist_moved", "thumb", "index", "ok", "palm_moved", "c"]
model = keras.models.load_model('handrecognition_model.h5')

# TensorFlow and tf.keras
#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras.preprocessing import image
#from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
# Recreate the exact same model, including weights and optimizer.
#import numpy as np
#import matplotlib.pyplot as plt
#import pandas as pd
path = "test_img3.png"
X = []  # Image data
#
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Converts into the corret colorspace (GRAY)
img = cv2.resize(img, (320, 120))  # Reduce image size so training can be faster
#
X.append(img)
X = np.array(X, dtype="float32")
#
X = X.reshape(1, 120, 320, 1)
# # test_image = test_image.reshape(img_width, img_height*3)    # Ambiguity!
# # Should this instead be: test_image.reshape(img_width, img_height, 3) ??
#
result = model.predict(X)
# # print("HI")
print(np.argmax(result))
#

text = 'Predicted: {} {:2.0f}% '.format(class_names[np.argmax(result)], 100*np.max(result))
#text = class_names[np.argmax(result)]
print(text)


img = cv2.imread(path)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Converts into the corret colorspace (GRAY)
#img = cv2.resize(img, (320, 120))  # Reduce image size so training can be faster
backSub = cv2.createBackgroundSubtractorKNN()
fgMask = backSub.apply(img)
cv2.imshow('Frame', img)
cv2.imshow('FG Mask', fgMask)
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (320, 120))
    #cv2.imshow("Image", img)
    fgMask = backSub.apply(img)
    cv2.imshow('Frame', img)
    cv2.imshow('FG Mask', fgMask)
    X = []  # Image data
    X.append(fgMask)
    X = np.array(X, dtype="float32")
    X = X.reshape(1, 120, 320, 1)
    result = model.predict(X)
    text = 'Predicted: {} {:2.0f}% '.format(class_names[np.argmax(result)], 100 * np.max(result))
    print(text)
    k = cv2.waitKey(1)
    if k == 27:
        break
#
# img = cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)