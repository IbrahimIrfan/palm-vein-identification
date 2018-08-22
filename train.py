import tensorflow as tf
from tensorflow import keras
from PIL import Image

import numpy as np

classes = ["left", "right"]

hand = "right"
pic = np.array(Image.open(hand + "Edited/thr" + str(0) + ".jpg"))
train_images = np.array([pic])
train_labels = np.array([1]*15)

for i in range(1, 15):
    pic = np.array(Image.open(hand + "Edited/thr" + str(i) + ".jpg"))
    train_images = np.vstack((train_images, np.array([pic])))

pic = np.array(Image.open(hand + "Edited/thr" + str(15) + ".jpg"))
test_images = np.array([pic])
test_labels = np.array([1]*5)

for i in range(16, 20):
    pic = np.array(Image.open(hand + "Edited/thr" + str(i) + ".jpg"))
    test_images = np.vstack((test_images, np.array([pic])))

print train_images.shape
print test_images.shape
