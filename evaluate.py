import os
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import math
import argparse
from glob import glob
from os import path
from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from config import EPOCHS, IMAGE_SIZE, BATCH_SIZE, ANNOTATIONS_ROOT, FRAMES_ROOT, MODEL_ROOT, CLASSES, VIDEOS_ROOT
from sklearn.metrics import accuracy_score
from scipy import stats as s

def evaluate_model(list_number = 1):
  print("👾 evaluating model on list number {}".format(list_number))
  print("👾 loading weights")
  model = keras.experimental.load_from_saved_model(path.join(MODEL_ROOT, "model.h5"))
  model.build(IMAGE_SIZE + (3,))
  print(model.summary())
  print("👾 reading test file")
  file = open(path.join(ANNOTATIONS_ROOT, "testlist0{}.txt".format(list_number)), "r")
  temp = file.read()
  videos = temp.split('\n')
  # pick up only the classes I need
  videos = list(set([a for a in videos for b in CLASSES if b in a.split("/")[0]]))

  # creating the dataframe
  print("👾 creating dataframe from test file")
  test = pd.DataFrame()
  test['video'] = videos
  test = test[:-1]
  test_videos = test['video']

  # creating the tags
  print("👾 generating tags for labels")
  train = pd.read_csv(path.join(ANNOTATIONS_ROOT, 'trainlist01_frames.csv'))
  y = train['label']
  y = pd.get_dummies(y)

  # creating two lists to store predicted and actual tags
  predict = []
  actual = []

  # checking if temp root exist otherwise create it
  if not path.exists("temp"):
    print("👾 creating folder temp")
    os.makedirs("temp")

  # for loop to extract frames from each test video
  print("👾 extracting frames from test videos")
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
      img = load_img(images[i], target_size = IMAGE_SIZE + (3,))
      img = img_to_array(img)
      prediction_images.append(img)
      
    # converting all the frames for a test video into numpy array
    prediction_images = np.array(prediction_images)
    # predicting tags for each array
    prediction = np.argmax(model.predict(prediction_images), axis=-1)
    print(prediction)
    # appending the mode of predictions in predict list to assign the tag to the video
    predict.append(y.columns.values[s.mode(prediction)[0][0]])
    # appending the actual tag of the video
    actual.append(videoFile.split('/')[0])
    
  print("evaluation done")
  print( accuracy_score(predict, actual) * 100 )

if __name__ == '__main__':
  # get arguments from terminal
  parser = argparse.ArgumentParser(description = "🤖 evaluate dataset on a test list")
  parser.add_argument(
    "--list_number",
    type = int,
    default = 1,
    help = "👾 test list number (1, 2, 3)"
  )
  args = parser.parse_args()
  evaluate_model(args.list_number)