from trafficsign.trafficsignmodel import TrafficSignNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from skimage import transform
from skimage import exposure
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input GTSRB")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
args = vars(ap.parse_args())

def load_split(basePath, csvPath):
	data = []
	labels = []
	rows = open(csvPath).read().strip().split("\n")[1:]
	random.shuffle(rows)
    
	for (i, row) in enumerate(rows):
		if i > 0 and i % 1000 == 0:
			print("Toplam {} adet goruntu islendi, lutfen bekleyiniz".format(i))
		(label, imagePath) = row.strip().split(",")[-2:]
		imagePath = os.path.sep.join([basePath, imagePath])
		image = io.imread(imagePath)
		image = transform.resize(image, (32, 32))
		image = exposure.equalize_adapthist(image, clip_limit=0.1)
		data.append(image)
		labels.append(int(label))
	data = np.array(data)
	labels = np.array(labels)
	return (data, labels)
NUM_EPOCHS = 12
learning_rate = 1e-3
Batch_Size = 64
labelNames = open("signnames.csv").read().strip().split("\n")[1:]
labelNames = [l.split(",")[1] for l in labelNames]
trainPath = os.path.sep.join([args["dataset"], "Train.csv"])
testPath = os.path.sep.join([args["dataset"], "Test.csv"])
print("egitim ve test verisi yukleniyor...")
(trainX, trainY) = load_split(args["dataset"], trainPath)
(testX, testY) = load_split(args["dataset"], testPath)
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0
numLabels = len(np.unique(trainY))
trainY = to_categorical(trainY, numLabels)
testY = to_categorical(testY, numLabels)

classTotals = trainY.sum(axis=0)
classWeight = classTotals.max() / classTotals


aug = ImageDataGenerator(
	rotation_range=10,
	zoom_range=0.15,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.15,
	horizontal_flip=False,
	vertical_flip=False,
	fill_mode="nearest")

print("model derleniyor...")
opt = Adam(lr=learning_rate, decay=learning_rate / (NUM_EPOCHS * 0.5))
model = TrafficSignNet.build(width=32, height=32, depth=3,classes=numLabels)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

print("model egitiliyor...")
H = model.fit_generator(
	aug.flow(trainX, trainY, batch_size=Batch_Size),
	validation_data=(testX, testY),
	steps_per_epoch=trainX.shape[0] // Batch_Size,
	epochs=NUM_EPOCHS,
	class_weight=classWeight,
	verbose=1)

print("model degerlendiriliyor...")
predictions = model.predict(testX, batch_size=Batch_Size)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=labelNames))
print("serializing network to '{}'...".format(args["model"]))
model.save(args["model"])
