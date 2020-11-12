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
from tensorflow.keras.layers import Conv2D
from sklearn.metrics import classification_report
import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy as np 
import argparse
import os 
# import gc

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot_v2.png", help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str, default="mask_detector_v2.model", help="path to mask detector model output")
args = vars(ap.parse_args())

# enable gpu
config = tf.compat.v1.ConfigProto(gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)) #device_count = {'GPU': 1}
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

print("[LOG] Checking GPU status...")

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

# initial learning rate, epochs, batch size
INIT_LR = 1e-4
EPOCHS = 5
BS = 32
SEED = 152

print("[LOG] Processing Images")

# get images from dataset directory
print("[LOG] Loading images...")

# train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
#     directory=args["dataset"],
#     labels="inferred",
#     label_mode="categorical",
#     class_names=["with_mask", "without_mask"],
#     batch_size=BS,
#     image_size=(224, 224),
#     seed=SEED,
#     validation_split=.2,
#     subset="training",
#     interpolation="bilinear",
#     follow_links=False,
# )

# valid_dataset = tf.keras.preprocessing.image_dataset_from_directory(
#     directory=args["dataset"],
#     labels="inferred",
#     label_mode="categorical",
#     class_names=["with_mask", "without_mask"],
#     batch_size=BS,
#     image_size=(224, 224),
#     seed=SEED,
#     validation_split=.2,
#     subset="validation",
#     interpolation="bilinear",
#     follow_links=False,
# )

# print(train_dataset)
# print(valid_dataset)

train_datagen = ImageDataGenerator(
    rotation_range=30, 
    zoom_range=0.2, 
    width_shift_range=0.2, 
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode="nearest",
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

valid_datagen = ImageDataGenerator(
    fill_mode="nearest",
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

print("[LOG] Initializing training set generator")
train_gen = train_datagen.flow_from_directory(
    args["dataset"],
    subset="training",
    shuffle=True,
    seed=SEED,
    target_size=(224, 224),
    batch_size=BS
)

print("[LOG] Initializing validation set generator")
valid_gen = valid_datagen.flow_from_directory(
    args["dataset"],
    subset="validation",
    shuffle=False,
    seed=SEED, 
    target_size=(224, 224),
    batch_size=BS
)

STEP_SIZE_TRAIN=train_gen.n//train_gen.batch_size
STEP_SIZE_VALID=valid_gen.n//valid_gen.batch_size

# print("\n[LOG] Print validation data classes\n")
# print(valid_gen.filenames)

print("[LOG] Generating Model...")
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# construct head of model that will be placed after the output layer
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)   # pool size = input shape dim / batch size
headModel = Conv2D(256, (1, 1), activation="relu")(headModel)
headModel = Flatten(name="flatten")(headModel)
# headModel = Dense(128, activation="relu")(headModel)
# headModel = Dropout(0.2)(headModel)                         # typical number for dropout
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# place head of FC model on top of base model (will become to model to train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over layers in base model and freeze such that they are not updated during training process
#Head layer weights will be tuned
for layer in baseModel.layers:
    layer.trainable = False

# compile model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
# model.summary()

# train head of network
print("[INFO] training head...")
H = model.fit(
    train_gen,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=valid_gen,
    validation_steps=STEP_SIZE_VALID,
    epochs=EPOCHS
)

# predict using testing set
print("[INFO] evaluating network...")
print(model.evaluate(valid_gen, steps=STEP_SIZE_VALID))

print("[INFO] predicting values....")
valid_gen.reset()
predIdxs = model.predict(valid_gen, verbose=1)
# for each image in testing set, find label with largest predicted probablity
predIdxs = predIdxs.argmax(axis=1)\

print("Dimension of predicted labels:", predIdxs.shape)
print("Dimension of validation labels:", valid_gen.classes.shape)

# classification report
print("[LOG] Printing classification report...")
print(classification_report(valid_gen.classes, predIdxs, target_names=["with_mask", "without_mask"]))

# serialize model to disk
print("[INFO] saving mask detector model:", args["model"])
model.save(args["model"], save_format="h5")


N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(1, N + 1), H.history["loss"], label="train_loss")
plt.plot(np.arange(1, N + 1), H.history["val_loss"], label="valid_loss")
plt.plot(np.arange(1, N + 1), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(1, N + 1), H.history["val_accuracy"], label="valid_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])