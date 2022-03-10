import os

# TensorFlow and tf.keras
#import tensorflow as tf
#from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten
# Recreate the exact same model, including weights and optimizer.
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from tensorflow.keras.preprocessing import image
# Sklearn
from sklearn.model_selection import train_test_split # Helps with organizing data for training
from sklearn.metrics import confusion_matrix # Helps present results as a confusion-matrix


hand_df = pd.read_csv("HandInfo.txt")
ImageFolder = "./Hands/"
SubImagePath = 'imageName'
LabelName = "aspectOfHand"
NbOfImages = len(hand_df)
print(len(hand_df))


#
# #We need to get all the paths for the images to later load them
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
X = []  # Image data
y = []  # Labels
class_names = ["dorsal right", "dorsal left", "palmar left", "palmar right"]
rotate = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE ]

Total = 0
# Loops through imagepaths to load images and labels into arrays
for i in range (0, NbOfImages):
    img = cv2.imread(ImageFolder + hand_df[SubImagePath][i])  # Reads image and returns np.array
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Converts into the corret colorspace (GRAY)
    img = cv2.resize(img, (320, 120))  # Reduce image size so training can be faster

    X.append(img)
    X.append(cv2.rotate(img, cv2.ROTATE_180))
   # X.append(cv2.rotate(img, cv2.ROTATE_180))
   # X.append(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))
    if hand_df[LabelName][i] == class_names[0]:
        y.append(0)
        y.append(0)
        # y.append(0)
        # y.append(0)
    elif hand_df[LabelName][i] == class_names[1]:
        y.append(1)
        y.append(1)
        # y.append(1)
        # y.append(1)
    elif hand_df[LabelName][i] == class_names[2]:
        y.append(2)
        y.append(2)
        # y.append(2)
        # y.append(2)
    elif hand_df[LabelName][i] == class_names[3]:
        y.append(3)
        y.append(3)
        # y.append(3)
        # y.append(3)
    #Total = Total + 1
    # Rotate image
    # for r in rotate:
    #     Total = Total + 1
    #     img = cv2.rotate(img, r)
    #     X.append(img)
    #     if hand_df[LabelName][i] == class_names[0]:
    #         y.append(0)
    #     elif hand_df[LabelName][i] == class_names[1]:
    #         y.append(1)
    #     elif hand_df[LabelName][i] == class_names[2]:
    #         y.append(2)
    #     elif hand_df[LabelName][i] == class_names[3]:
    #         y.append(3)

# Turn X and y into np.array to speed up train_test_split
X = np.array(X, dtype="uint8")
X = X.reshape(NbOfImages*2, 120, 320, 1)  # Needed to reshape so CNN knows it's different images
y = np.array(y)

print("Images loaded: ", len(X))
print("Labels loaded: ", len(y))
#
# print(y[0], imagepaths[0])  # Debugging
#
ts = 0.3 # Percentage of images that we want to use for testing. The rest is used for training.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=42)

model = Sequential()
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(120, 320, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))


# Configures the model for training
model.compile(optimizer='RMSprop', # Optimization routine, which tells the computer how to adjust the parameter values to minimize the loss function.
              loss='sparse_categorical_crossentropy', # Loss function, which tells us how bad our predictions are.
              metrics=['accuracy']) # List of metrics to be evaluated by the model during training and testing.



# Trains the model for a given number of epochs (iterations on a dataset) and validates it.
model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=2, validation_data=(X_test, y_test))

# Save entire model to a HDF5 file
model.save('handSide_model.h5')

predictions = model.predict(X_test) # Make predictions towards the test set

np.argmax(predictions[0]), y_test[0] # If same, got it right


# Function to plot images and labels for validation purposes
def validate_9_images(predictions_array, true_label_array, img_array):
  # Array for pretty printing and then figure size

  plt.figure(figsize=(15, 5))

  for i in range(1, 10):
    # Just assigning variables
    prediction = predictions_array[i]
    true_label = true_label_array[i]
    img = img_array[i]
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Plot in a good way
    plt.subplot(3, 3, i)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(prediction)  # Get index of the predicted label from prediction

    # Change color of title based on good prediction or not
    if predicted_label == true_label:
      color = 'blue'
    else:
      color = 'red'

    plt.xlabel("Predicted: {} {:2.0f}% (True: {})".format(class_names[predicted_label],
                                                          100 * np.max(prediction),
                                                          class_names[true_label]),
               color=color)
  plt.show()

validate_9_images(predictions, y_test, X_test)