import tensorflow as tf
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import matplotlib




'''

'''


IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3


DATA_PATH = 'New folder/'


path, dirs, files = next(os.walk(DATA_PATH))
images_number = len(files)
print("\n")
print("FOUND: ", images_number, "images")
print("\n")


# _______________________________  D A T A   R E A D  _________________________________
k = 0
for item in os.listdir(DATA_PATH):
    data = cv2.imread(os.path.join(DATA_PATH, item))
    data = cv2.resize(data, (IMG_WIDTH, IMG_HEIGHT))
    # Convert BGR to RGB
    data_RGB = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

    mask_RGB = data_RGB
    plt.imshow(mask_RGB)
    plt.show()
    mask_synthesis = np.zeros((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    # Convert BGR to HSV
    mask_HSV = cv2.cvtColor(mask_RGB, cv2.COLOR_RGB2HSV)

    low_bound = np.array([21, 27, 121])
    up_bound = np.array([70, 80, 200])
    temp = cv2.inRange(mask_HSV, low_bound, up_bound)
    for i in range(IMG_HEIGHT):
        for j in range(IMG_WIDTH):
            if temp[i][j] == 255:
                mask_synthesis[i][j] = np.array([255, 255, 255])

    f = plt.figure()
    f.add_subplot(1, 2, 1)
    plt.imshow(np.rot90(data_RGB, 2))
    f.add_subplot(1, 2, 2)
    plt.imshow(np.rot90(mask_synthesis, 2))
    plt.show(block=True)

    print(mask_synthesis)

    savepath = 'New folder/'
    # cv2.imwrite(os.path.join(savepath, item), cv2.cvtColor(mask_synthesis, cv2.COLOR_RGB2BGR))

    k = k+1

