#imports
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt 
import numpy as np 
import argparse
import os 

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str, default="mask_detector.model", help="path to mask detector model output")
args = vars(ap.parse_args())

# initial learning rate, epochs, batch size
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

#get images from dataset directory
print("Loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

#loop over image paths
for imagePath in imagePaths:
    #get label for image
    label = imagePath.split(os.path.sep)[-3]
    #load image and preprocess
    image = load_img(imagePath, target_size=(128, 128))
    image = img_to_array(image)
    image = preprocess_input(image)

    #append image/ label to list
    data.append(image)
    labels.append(label)

#convert data and labels to numpy array
data = np.array(data, dtype="float32")
labels = np.array(labels)

print("[LOG] Done loading.")

#one-hot encoding on label (does it have to be one hot)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

#partition data into training/testing splits
(trainData, testData, trainLabels, testLabels) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)
print("[LOG] Print arary shape")
print(trainData.shape, "\n")
print(trainLabels.shape, "\n")

#data augmentation parameters: randomly zoom, shear, rotate, shift and flip images
aug = ImageDataGenerator(
    rotation_range=20, 
    zoom_range=0.15, 
    width_shift_range=0.2, 
    height_shift_range=0.2,
    shear_range=0.15, 
    horizontal_flip=True,
    fill_mode="nearest"
)

# load MobileNetV2
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=[128, 128, 3]))

#construct head of model that will be placed on top of the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(8, 8))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

#place head of FC model on top of base model (will become to model to train)
model = Model(inputs=baseModel.input, outputs=headModel)

#loop over layers in base model and freeze such that they are not updated during training process
#Head layer weights will be tuned
for layer in baseModel.layers:
    layer.trainable = False

#compile model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

#train head of network
print("[INFO] training head...")
H = model.fit(
    aug.flow(trainData, trainLabels, bat_size=BS),
    steps_per_epoch=len(trainData) // BS,
    validation_data = (testData, testLabels),
    validation_steps = len(trainData) // BS,
    epochs=EPOCHS
)

#predict using testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testData, batch_size=BS)

#for each image in testing set, find label with largest predicted probablity
predIdxs = np.argmax(predIdxs, axis=1)

#classification report
print(classification_report(testLabels.argmax(axis=1), predIdxs, target_names=lb.classes_))

#serialize model to disk
print("[INFO] saving mask detector model")
model.save(args["model"], save_format="h5")

#plot accuracy/loss curve
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legent(loc="lower left")
plt.savefig(args["plot"])