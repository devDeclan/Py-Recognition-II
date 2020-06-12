import os
import argparse
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import math
from os import path
from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from config import EPOCHS, IMAGE_SIZE, BATCH_SIZE, ANNOTATIONS_ROOT, FRAMES_ROOT, MODEL_ROOT, CLASSES, VIDEOS_ROOT
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy import stats as s

def load_data(list_number):
  print("👾 reading training frames")
  dataset = pd.read_csv(path.join(ANNOTATIONS_ROOT, "trainlist0{}_frames.csv".format(list_number)))
  print(dataset.head())
  print(dataset.tail())
  
  print("👾 loading training frames")
  images = []
  for i in tqdm(range(dataset.shape[0])):
    image = load_img(dataset["image"][i], target_size = IMAGE_SIZE)
    image = img_to_array(image)
    images.append(image)
  X = np.array(images)
  y = dataset["label"]

  print("👾 generating train and test splits")
  X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify = y)

  y_train = pd.get_dummies(y_train)
  y_test = pd.get_dummies(y_test)
  
  return X_train, y_train, X_test, y_test

def make_model(input_shape, num_classes):
  data_augmentation = keras.Sequential(
    [
      layers.experimental.preprocessing.RandomFlip("horizontal"),
      layers.experimental.preprocessing.RandomRotation(0.1),
    ]
  )
  inputs = keras.Input(shape=input_shape)
  # Image augmentation block
  x = data_augmentation(inputs)

  # Entry block
  x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
  x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
  x = layers.BatchNormalization()(x)
  x = layers.Activation("relu")(x)

  x = layers.Conv2D(64, 3, padding="same")(x)
  x = layers.BatchNormalization()(x)
  x = layers.Activation("relu")(x)

  previous_block_activation = x  # Set aside residual

  # had to use a for loop, can't type it over and over again
  for size in [128, 256, 512, 728]:
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(size, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(size, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    # Project residual
    residual = layers.Conv2D(size, 1, strides=2, padding="same")(previous_block_activation)
    x = layers.add([x, residual])  # Add back residual
    previous_block_activation = x  # Set aside next residual

  x = layers.SeparableConv2D(1024, 3, padding="same")(x)
  x = layers.BatchNormalization()(x)
  x = layers.Activation("relu")(x)

  # using softmax here cos Im working with multiple classes
  x = layers.GlobalAveragePooling2D()(x)
  activation = "softmax"
  units = num_classes

  # add dropouts to prevent overfitting
  x = layers.Dropout(0.5)(x)
  outputs = layers.Dense(units, activation=activation)(x)
  return keras.Model(inputs, outputs)

def main(list_number = 1):
  # loading the data
  print("🤖 loading data from train list {}".format(list_number))
  X_train, y_train, X_test, y_test = load_data(list_number)

  # make model
  print("🤖 defining my model")
  model = make_model(input_shape=IMAGE_SIZE + (3,), num_classes=len(CLASSES))
  # keras.utils.plot_model(model, show_shapes=True)
  print(model.summary())
  
  # checking if models root exist otherwise create it
  if not path.exists(MODEL_ROOT):
    print("👾 creating folder {}".format(MODEL_ROOT))
    os.makedirs(MODEL_ROOT)

  # start training
  print("🤖 training started")
  callbacks = [
    keras.callbacks.ModelCheckpoint("model/save_at_{epoch:02d}.h5"),
  ]
  model.compile(
    optimizer = keras.optimizers.Adam(1e-3),
    loss = "categorical_crossentropy",
    metrics = ["accuracy"],
  )
  print(X_train.shape)
  print(y_train.shape)
  print(y_train)
  print(X_test.shape)
  print(y_test.shape)
  print(y_test)

  model.fit(
    X_train,
    y_train,
    epochs = EPOCHS,
    callbacks = callbacks,
    validation_data = (
      X_test,
      y_test
    ),
    batch_size = BATCH_SIZE
  )

  # save model
  "🤖 saving model"
  tf.compat.v1.keras.experimental.export_saved_model(model, path.join(MODEL_ROOT, "model.h5"))

  # convert model to json
  "🤖 saving model to json"
  json_model = model.to_json()
  jsonfile = open(path.join(MODEL_ROOT, "model.json"), "w")
  jsonfile.write(json_model)
  jsonfile.close()

if __name__ == '__main__':
  # get arguments from terminal
  parser = argparse.ArgumentParser(description = "🤖 train dataset on a train list")
  parser.add_argument(
    "--list_number",
    type = int,
    default = 1,
    help = "👾 train list number (1, 2, 3)"
  )
  args = parser.parse_args()
  main(args.list_number)