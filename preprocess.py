import os
import sys
import argparse
import glob
import cv2
import random
import math
import fnmatch
import pandas as pd
import numpy as np
import tensorflow as tf
from os import path
from tqdm import tqdm
from config import DATASET_ROOT, WORKERS, VIDEOS_ROOT, FRAMES_ROOT, ANNOTATIONS_ROOT, CLASSES
from multiprocessing import Pool, current_process

def decode_video_to_frames(video_item):
  full_path, video_path, video_id = video_item
  video_name = video_path.split(".")[0]
  out_full_path = path.join(FRAMES_ROOT, video_name)
  try:
    os.mkdir(out_full_path)
  except OSError:
    pass
  cap = cv2.VideoCapture(full_path)   # capturing the video from the given path
  frameRate = cap.get(5) #frame rate
  count = 1
  while(cap.isOpened()):
      frameId = cap.get(1) #current frame number
      ret, frame = cap.read()
      if (ret != True):
        break
      if (frameId % math.floor(frameRate) == 0):
        cv2.imwrite(
          "{}/img_{:05d}.jpg".format(
            out_full_path,
            count
          ),
          frame
        )
      count += 1
  cap.release()
  print("🤓 {} done with {} frames".format(video_name, count))
  sys.stdout.flush()
  return True

def decode_videos_to_frames():
  # checking if frames root exist otherwise create it
  if not path.exists(FRAMES_ROOT):
    print("👾 creating folder {}".format(FRAMES_ROOT))
    os.makedirs(FRAMES_ROOT)

  # create sub directories for frame classes
  videos_path = path.join(DATASET_ROOT, "videos")
  classes = os.listdir(videos_path)
  for classname in classes:
    class_dir = path.join(FRAMES_ROOT, classname)
    if not path.isdir(class_dir):
      print("    👾 creating folder {}".format(class_dir))
      os.makedirs(class_dir)

  # read videos from folder
  fullpath_list = glob.glob(videos_path + "/*/*.avi")
  done_fullpath_list = glob.glob(FRAMES_ROOT + "/*/*")
  print("👾 total number of videos {}".format(len(fullpath_list)))
  print("👾 total number of decoded videos {}".format(len(done_fullpath_list)))
  fullpath_list = set(fullpath_list).difference(set(done_fullpath_list))
  print("👾 total number of undecoded videos {}".format(len(fullpath_list)))
  fullpath_list = list(fullpath_list)
  
  #
  video_list = list(
    map(
      lambda p: path.join("/".join(p.split("/")[-2:])),
      fullpath_list
    )
  )

  pool = Pool(WORKERS)
  pool.map(
    decode_video_to_frames,
    zip(
      fullpath_list,
      video_list,
      range(len(video_list))
    )
  )

def build_list():
  splits = glob.glob("{}/trainlist*.txt".format(ANNOTATIONS_ROOT))
  print(splits)
  for i in tqdm(range(len(splits))):
    file = open(splits[i], "r")
    temp = file.read()
    videos = temp.split("\n")
    videos = list(set([a for a in videos for b in CLASSES if b == a.split("/")[0]]))
    print(len(videos))

    print("👾 obtaining frames")
    frames_list = []
    for video in tqdm(range(len(videos))):
      frames = glob.glob(
        "{}/{}/*.jpg".format(
          FRAMES_ROOT,
          videos[video].split(".")[0]
        )
      )
      frames_list.extend(frames)
    df = pd.DataFrame()
    df['image'] = frames_list
    df = df[:-1]
    print("👾 frames obtained")

    print("👾 adding labels")
    labels_list = []
    for frame in tqdm(range(df.shape[0])):
      labels_list.append(df['image'][frame].split('/')[2])
        
    df['label'] = labels_list
    print("👾 labels added")

    filename = "{}_frames.csv".format(
      splits[i].split(".")[0]
    )
    print("👾 saving to file {}".format(filename))
    df.to_csv(filename, index = False)
    print("👾 file saved")  

def clear_corrupted():
  print("👾 fishing out corrupted images")  
  num_skipped = 0
  for folder_name in CLASSES:
    folder_path = path.join(FRAMES_ROOT, folder_name)
    for sub_folder_name in os.listdir(folder_path):
      sub_folder_path = path.join(folder_path, sub_folder_name)
      for fname in os.listdir(sub_folder_path):
        fpath = path.join(sub_folder_path, fname)
        try:
          fobj = open(fpath, "rb")
          is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
        finally:
          fobj.close()

        if not is_jfif:
          num_skipped += 1
          # Delete corrupted image
          os.remove(fpath)
  print("👾 deleted {} images".format(num_skipped))

if __name__ == '__main__':
  # get arguments from terminal
  parser = argparse.ArgumentParser(description = "🤖 prepare dataset for training")
  parser.add_argument(
    "--decode_video",
    action = "store_true",
    default = True,
    help = "👾 generate frames from video data"
  )
  parser.add_argument(
    "--clear_corrupted",
    action = "store_true",
    default = True,
    help = "👾 clear corrupted video frames"
  )
  parser.add_argument(
    "--build_list",
    action = "store_true",
    default = True,
    help = "👾 generate csv file for training"
  )
  args = parser.parse_args()

  if args.decode_video:
    print("🤖 decoding videos to frames")
    decode_videos_to_frames()
  if args.clear_corrupted:
    print("🤖 clearing currupted frames")
    clear_corrupted()
  if args.build_list:
    print("🤖 generating training files")
    build_list()