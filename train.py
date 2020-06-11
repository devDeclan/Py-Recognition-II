import os
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
from lib.config import EPOCHS, IMAGE_SIZE, BATCH_SIZE, ANNOTATIONS_ROOT, FRAMES_ROOT, MODEL_ROOT, CLASSES, VIDEOS_ROOT
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy import stats as s
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
    dataset = pd.read_csv(path.join(ANNOTATIONS_ROOT, 'trainlist01_frames.csv'))
    print(dataset.head())
    print(dataset.tail())
    
    images = []
    for i in tqdm(range(dataset.shape[0])):
        image = load_img(dataset["image"][i], target_size = IMAGE_SIZE)
        image = img_to_array(image)
        images.append(image)
    X = np.array(images)
    y = dataset["label"]
    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify = y)

    y_train = pd.get_dummies(y_train)
    y_test = pd.get_dummies(y_test)
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
    activation = "softmax"
    units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)

def evaluate_model():

    model = make_model(input_shape=IMAGE_SIZE + (3,), num_classes=len(CLASSES))
    model.load_weights(path.join(MODEL_ROOT, "model.hdf5"))
    model.compile(
        optimizer = keras.optimizers.Adam(1e-3),
        loss = "categorical_crossentropy",
        metrics = ["accuracy"],
    )
    file = open("testlist01.txt", "r")
    temp = file.read()
    videos = temp.split('\n')
    # pick up only the classes I need
    videos = list(set([a for a in videos for b in CLASSES if b in a.split("/")[0]]))

    # creating the dataframe
    test = pd.DataFrame()
    test['video'] = videos
    test = test[:-1]
    test_videos = test['video']

    # creating the tags
    train = pd.read_csv(path.join(ANNOTATIONS_ROOT, 'trainlist01_frames.csv'))
    y = train['label']
    y = pd.get_dummies(y)

    # creating two lists to store predicted and actual tags
    predict = []
    actual = []

    # for loop to extract frames from each test video
    for i in tqdm(range(test_videos.shape[0])):
        count = 0
        videoFile = test_videos[i]
        cap = cv2.VideoCapture(path.join(VIDEOS_ROOT, videoFile))   # capturing the video from the given path
        frameRate = cap.get(5) #frame rate
        x=1
        # removing all other files from the temp folder
        files = glob('temp/*')
        for f in files:
            os.remove(f)
        while(cap.isOpened()):
            frameId = cap.get(1) #current frame number
            ret, frame = cap.read()
            if (ret != True):
                break
            if (frameId % math.floor(frameRate) == 0):
                # storing the frames of this particular video in temp folder
                filename ='temp/' + "_frame%d.jpg" % count
                count+=1
                cv2.imwrite(filename, frame)
        cap.release()
        
        # reading all the frames from temp folder
        images = glob("temp/*.jpg")
        
        prediction_images = []
        for i in range(len(images)):
            img = image.load_img(images[i], target_size = IMAGE_SIZE + (3,))
            img = image.img_to_array(img)
            prediction_images.append(img)
            
        # converting all the frames for a test video into numpy array
        prediction_images = np.array(prediction_images)
        # predicting tags for each array
        prediction = model.predict_classes(prediction_images)
        # appending the mode of predictions in predict list to assign the tag to the video
        predict.append(y.columns.values[s.mode(prediction)[0][0]])
        # appending the actual tag of the video
        actual.append(videoFile.split('/')[0])

        accuracy_score(predict, actual)*100

def main():
    # loading the data
    X_train, y_train, X_test, y_test = load_data()

    # make model
    model = make_model(input_shape=IMAGE_SIZE + (3,), num_classes=len(CLASSES))
    # keras.utils.plot_model(model, show_shapes=True)
    print(model.summary())
    
    # checking if models root exist otherwise create it
    if not path.exists(MODEL_ROOT):
        print("👾 creating folder {}".format(MODEL_ROOT))
        os.makedirs(MODEL_ROOT)

    # start training
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
    model.save(path.join(MODEL_ROOT, "model.hdf5"))

    #convert model to json
    json_model = model.to_json()
    jsonfile = open(path.join(MODEL_ROOT, "model.json"), "w")
    jsonfile.write(json_model)
    jsonfile.close()

    # run inference
    evaluate_model()

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