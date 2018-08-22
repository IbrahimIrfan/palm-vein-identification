import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import os
import time

classes = ["left", "right"]

num_right_train = 15
num_left_train = 15

hand = "right"
pic = np.array(Image.open(hand + "Edited/thr" + str(0) + ".jpg"))
train_images = np.array([pic])
train_labels = np.array([1]*num_right_train + [0]*num_left_train)

for i in range(1, num_right_train):
    pic = np.array(Image.open(hand + "Edited/thr" + str(i) + ".jpg"))
    train_images = np.vstack((train_images, np.array([pic])))

hand = "left"
for i in range(num_left_train):
    pic = np.array(Image.open(hand + "Edited/thr" + str(i) + ".jpg"))
    train_images = np.vstack((train_images, np.array([pic])))

train_images = train_images / 255.0
print train_labels

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(600, 600)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(), 
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

while raw_input("photo? ") == "y":
    print "about to take photo"
    cmd = "raspistill -vf -w 600 -h 600 -roi 0.46,0.34,0.25,0.25 -o test/pic.jpg"
    os.system(cmd)
    img = cv2.imread("test/pic.jpg")
    # noise
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noise = cv2.fastNlMeansDenoising(gray)
    noise = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)

    # equalist hist
    kernel = np.ones((7,7),np.uint8)
    img = cv2.morphologyEx(noise, cv2.MORPH_OPEN, kernel)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    # invert
    inv = cv2.bitwise_not(img_output)

    # erode
    gray = cv2.cvtColor(inv, cv2.COLOR_BGR2GRAY)
    erosion = cv2.erode(gray,kernel,iterations = 1)

    # skel
    img = gray.copy()
    skel = img.copy()
    skel[:,:] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
    iterations = 0

    while True:
        eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        temp  = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img[:,:] = eroded[:,:]
        if cv2.countNonZero(img) == 0:
            break

    ret, thr = cv2.threshold(skel, 5,255, cv2.THRESH_BINARY);

    cv2.imwrite("test/thr.jpg", thr)

    pic = np.array(Image.open("test/thr.jpg"))
    test_images = np.array([pic])
    predictions = model.predict(test_images)
    print "final answer:"
    print classes[np.argmax(predictions[0])]
