import os
import tensorflow as tf
import pandas as pd
import numpy as np
from os import path
from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from lib.config import EPOCHS, IMAGE_SIZE, BATCH_SIZE, ANNOTATIONS_ROOT, FRAMES_ROOT
print(tf.version.VERSION)

def clear_corrupted():
	num_skipped = 0
	for folder_name in ("Cat", "Dog"):
		folder_path = os.path.join("PetImages", folder_name)
		for fname in os.listdir(folder_path):
			fpath = os.path.join(folder_path, fname)
			try:
				fobj = open(fpath, "rb")
				is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
			finally:
				fobj.close()

			if not is_jfif:
				num_skipped += 1
				# Delete corrupted image
				os.remove(fpath)
	print("Deleted %d images" % num_skipped)

def load_data():
	train_dataset = pd.read_csv(path.join(ANNOTATIONS_ROOT, 'trainlist01_frames.csv'))
	test_dataset = pd.read_csv(path.join(ANNOTATIONS_ROOT, "testlist01_frames.csv"))
	print(train_dataset.head())
	print(test_dataset.head())
	
	train_images = []
	for i in tqdm(range(train_dataset.shape[0])):
		image = load_img(train_dataset["image"][i], target_size = IMAGE_SIZE)
		image = img_to_array(image)
		train_images.append(image)
	X_train = np.array(train_images)
	y_train = train_dataset["label"]


	test_images = []
	for i in tqdm(range(test_dataset.shape[0])):
		image = load_img(test_dataset["image"][i], target_size = IMAGE_SIZE)
		image = img_to_array(image)
		test_images.append(image)
	X_test = np.array(test_images)
	y_test = test_dataset["label"]

	#y_train = pd.get_dummies(y_train)
#	y_test = pd.get_dummies(y_test)
	print(X_train.shape)
	print(y_train.shape)
	
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

	x = layers.GlobalAveragePooling2D()(x)
	if num_classes == 2:
		activation = "sigmoid"
		units = 1
	else:
		activation = "softmax"
		units = num_classes

	x = layers.Dropout(0.5)(x)
	outputs = layers.Dense(units, activation=activation)(x)
	return keras.Model(inputs, outputs)


def main():
	# loading the data
	X_train, y_train, X_test, y_test = load_data()

	# make model
	model = make_model(input_shape=IMAGE_SIZE + (3,), num_classes=101)
	# keras.utils.plot_model(model, show_shapes=True)
	print(model.summary())

	# start training
	callbacks = [
		keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
	]
	model.compile(
		optimizer = keras.optimizers.Adam(1e-3),
		loss = "categorical_crossentropy",
		metrics = ["accuracy"],
	)
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
	model.save(MODEL_ROOT)

	# run inference
	'''img = keras.preprocessing.image.load_img(
    "PetImages/Cat/6779.jpg", target_size=IMAGE_SIZE
	)
	img_array = keras.preprocessing.image.img_to_array(img)
	img_array = tf.expand_dims(img_array, 0)  # Create batch axis

	predictions = model.predict(img_array)
	score = predictions[0]
	print(
	    "This image is %.2f percent cat and %.2f percent dog."
	    % (100 * (1 - score), 100 * score)
	)'''

if __name__ == '__main__':
	main()