import tensorflow as tf
from tensorflow import keras
from PIL import Image

import numpy as np

classes = ["left", "right"]

hand = "right"
train_images = np.array([])
train_labels = np.array([1]*15)

for i in range(15):
    pic = Image.open(hand + "Edited/thr" + str(i) + ".jpg")
    print np.array(pic)
    #train_images = np.append(train_images, np.array(pic))

test_images = np.array([])
test_labels = np.array([1]*5)

for i in range(15, 20):
    pic = Image.open(hand + "Edited/thr" + str(i) + ".jpg")
    print np.array(pic)
    #test_images = np.append(test_images, np.array(pic))

print train_images.shape
print test_images.shape
