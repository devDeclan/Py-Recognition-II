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
    # run inference
    evaluate_model()

if __name__ == '__main__':
    main()