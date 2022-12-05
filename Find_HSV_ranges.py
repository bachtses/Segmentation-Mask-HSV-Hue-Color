# finding hsv range of target object(pen)
import cv2
import numpy as np
import time
import tensorflow as tf
import os
import cv2
import sys
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import matplotlib
from itertools import chain
from keras.models import model_from_json
import json


# A required callback method that goes into the trackbar function.
def nothing(x):
    pass


# Initializing the webcam feed.
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Create a window named trackbars.
cv2.namedWindow("Trackbars")

# Now create 6 trackbars that will control the lower and upper range of
# H,S and V channels. The Arguments are like this: Name of trackbar,
# window name, range,callback function. For Hue the range is 0-179 and
# for S,V its 0-255.
cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)


IMG_HEIGHT = 300
IMG_WIDTH = 300
IMG_CHANNELS = 3


DATA_PATH = 'New folder/'


path, dirs, files = next(os.walk(DATA_PATH))
images_number = len(files)
print("\n")
print("FOUND: ", images_number, "images")
print("\n")


# _______________________________  D A T A   R E A D  _________________________________
k = 0
while 1:
    for item in os.listdir(DATA_PATH):
        data = cv2.imread(os.path.join(DATA_PATH, item))
        data = cv2.resize(data, (IMG_WIDTH * 2, IMG_HEIGHT))
        ret = False
        # Convert the BGR image to HSV image.
        hsv = cv2.cvtColor(data, cv2.COLOR_BGR2HSV)

        # Get the new values of the trackbar in real time as the user changes
        # them
        l_h = cv2.getTrackbarPos("L - H", "Trackbars")
        l_s = cv2.getTrackbarPos("L - S", "Trackbars")
        l_v = cv2.getTrackbarPos("L - V", "Trackbars")
        u_h = cv2.getTrackbarPos("U - H", "Trackbars")
        u_s = cv2.getTrackbarPos("U - S", "Trackbars")
        u_v = cv2.getTrackbarPos("U - V", "Trackbars")

        # Set the lower and upper HSV range according to the value selected
        # by the trackbar
        lower_range = np.array([l_h, l_s, l_v])
        upper_range = np.array([u_h, u_s, u_v])

        # Filter the image and get the binary mask, where white represents
        # your target color
        mask = cv2.inRange(hsv, lower_range, upper_range)

        # You can also visualize the real part of the target color (Optional)
        res = cv2.bitwise_and(data, data, mask=mask)

        # Converting the binary mask to 3 channel image, this is just so
        # we can stack it with the others
        mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # stack the mask, orginal frame and the filtered result
        stacked = np.hstack((mask_3, data, res))

        # Show this stacked frame at 40% of the size.
        cv2.imshow('Trackbars', cv2.resize(stacked, None, fx=0.7, fy=1))

        # If the user presses ESC then exit the program
        key = cv2.waitKey(1)
        if key == 27:
            break


# Release the camera & destroy the windows.
cap.release()
cv2.destroyAllWindows()
