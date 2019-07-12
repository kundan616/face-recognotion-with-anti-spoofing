import matplotlib
matplotlib.use("Agg")

from architecture_definition.spoofproof import AntiSpoof
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import np_utils
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os

print("""\n\nMajor Project - Face Recognition using Siamese Network (anti-spoofing network training)

by - Abhishek Mann, Kundan Sharma
""")

initial_lr = 1e-4
batch_size = 8
epochs = 65

image_paths = list(paths.list_images("anti_spoof_dataset"))
data = []
labels = []
print("||Standby|| loading images from anti_spoof_dataset")

for image_path in image_paths:
    label = image_path.split(os.path.sep)[-2]
    img = cv2.resize(cv2.imread(image_path), (32,32))

    data.append(img)
    labels.append(label)

data = np.array(data, dtype = "float")/255.0

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
labels = np_utils.to_categorical(labels, 2)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0.3, random_state = 1)
augment = ImageDataGenerator(rotation_range = 20, zoom_range = 0.15, width_shift_range = 0.2, height_shift_range = 0.2, shear_range = 0.15, horizontal_flip = True, fill_mode = "nearest")

print("||Standby|| model compilation")
optimizer_algo = Adam(lr = initial_lr, decay = initial_lr/(epochs))
model = AntiSpoof.build(width = 32, height = 32, depth = 3, classes = len(label_encoder.classes_))
model.compile(loss="categorical_crossentropy", optimizer = optimizer_algo, metrics = ["accuracy"])

print("||Standby|| training network for epochs --> {}".format(epochs))
model_specs = model.fit_generator(augment.flow(trainX, trainY, batch_size = batch_size), validation_data = (testX, testY), steps_per_epoch = len(trainX)//batch_size, epochs = epochs)

print("||Standby|| Evaluating the network: ")
predics = model.predict(testX, batch_size = batch_size)
print(classification_report(testY.argmax(axis = 1), predics.argmax(axis = 1), target_names = label_encoder.classes_))

print("||Standby|| saving model to disk --> {}".format("anti_spoof_model.model"))
model.save("anti_spoof_model.model")

f = open("label_encoded.pickle", "wb")
f.write(pickle.dumps(label_encoder))
f.close()

plt.style.use("seaborn-bright")
plt.figure()
plt.plot(np.arange(0, epochs), model_specs.history["loss"], label = "training_set_loss")
plt.plot(np.arange(0, epochs), model_specs.history["val_loss"], label = "validation_set_loss")
plt.title("Training and cross-validation loss (Learning Curve 1)")
plt.xlabel("no. of epochs")
plt.ylabel("loss")
plt.legend(loc = "upper right")
plt.savefig("loss_curve.png")

plt.style.use("seaborn-bright")
plt.figure()
plt.plot(np.arange(0, epochs), model_specs.history["acc"], label = "training_set_accuracy")
plt.plot(np.arange(0, epochs), model_specs.history["val_acc"], label = "validation_set_accuracy")
plt.title("Training and cross-validation accuracy (Learning Curve 2)")
plt.xlabel("no. of epochs")
plt.ylabel("accuracy")
plt.legend(loc = "lower right")
plt.savefig("accuracy_curve.png")
