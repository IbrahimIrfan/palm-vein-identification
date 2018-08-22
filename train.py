import tensorflow as tf
from tensorflow import keras
from PIL import Image

import numpy as np

classes = ["left", "right"]

hand = "right"
pic = np.array(Image.open(hand + "Edited/thr" + str(0) + ".jpg"))
train_images = np.array([pic])
train_labels = np.array([1]*15 + [0]*15)

for i in range(1, 15):
    pic = np.array(Image.open(hand + "Edited/thr" + str(i) + ".jpg"))
    train_images = np.vstack((train_images, np.array([pic])))

pic = np.array(Image.open(hand + "Edited/thr" + str(15) + ".jpg"))
test_images = np.array([pic])
test_labels = np.array([1]*5 + [0]*5)

for i in range(16, 20):
    pic = np.array(Image.open(hand + "Edited/thr" + str(i) + ".jpg"))
    test_images = np.vstack((test_images, np.array([pic])))

hand = "left"
for i in range(15):
    pic = np.array(Image.open(hand + "Edited/thr" + str(i) + ".jpg"))
    train_images = np.vstack((train_images, np.array([pic])))

for i in range(15, 20):
    pic = np.array(Image.open(hand + "Edited/thr" + str(i) + ".jpg"))
    test_images = np.vstack((test_images, np.array([pic])))

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(600, 600)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(), 
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

predictions = model.predict(test_images)
for i in range(len(predictions)):
    print (classes[np.argmax(predictions[i])], classes[test_labels[i]])
