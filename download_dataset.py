import os
import sys
import argparse
import glob
import mmcv
import random
import fnmatch
import pandas as pd
from os import path
from tqdm import tqdm
from lib.config import DATASET_ROOT, WORKERS, VIDEOS_ROOT, FRAMES_ROOT, ANNOTATIONS_ROOT
from multiprocessing import Pool, current_process

def download_dataset():
  # checking if dataset root exist otherwise create it
  if not path.exists(DATASET_ROOT):
    print("👾 creating folder {}".format(DATASET_ROOT))
    os.makedirs(DATASET_ROOT)

  # check if the dataset file has already been downloaded
  # otherwise download it
  # (ano get data like that😂)
  dataset_rar = path.join(DATASET_ROOT, "dataset.rar")
  if not path.exists(dataset_rar):
    print("👾 downloading dataset file")
    dataset_url = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
    command = " ".join([
      "wget", "--quiet",
      "--directory-prefix", DATASET_ROOT,
      "--output-document", dataset_rar,
      dataset_url
    ])
    os.system(command)
    print("🤓 dataset file downloaded")
  else:
    print("👾 dataset file already exists, skipping download")

  # extract the videos
  print("👾 extracting dataset file")
  command = " ".join([
    "unrar", "x",
    dataset_rar, DATASET_ROOT,
    ">", "/dev/null",
    "&&", "mv",
    path.join(DATASET_ROOT, "UCF-101"),
    VIDEOS_ROOT
  ])
  os.system(command)
  print("🤓 dataset file downloaded and extracted")

  # check if the annotations file has already been download
  # otherwise download it
  annotations_zip = path.join(DATASET_ROOT, "annotations.zip")
  if not path.exists(annotations_zip):
    print("👾 downloading annotations file")
    annotations_url = "https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip"
    command = " ".join([
      "wget", "--quiet",
      "--directory-prefix", DATASET_ROOT,
      "--output-document", annotations_zip,
      annotations_url
    ])
    os.system(command)
    print("🤓 annotations file downloaded")
  else:
    print("👾 annotations file already exists, skipping download")

  # extract the annotations
  print("👾 extracting annotations file")
  command = " ".join([
    "unzip",
    "-q", annotations_zip,
    "-d", DATASET_ROOT,
    "&&", "mv",
    path.join(DATASET_ROOT, "ucfTrainTestlist"),
    ANNOTATIONS_ROOT
  ])
  os.system(command)
  print("🤓 annotations file downloaded and extracted")

def decode_video_to_frames(video_item):
  full_path, video_path, video_id = video_item
  video_name = video_path.split(".")[0]
  out_full_path = path.join(FRAMES_ROOT, video_name)
  try:
    os.mkdir(out_full_path)
  except OSError:
    pass
  video = mmcv.VideoReader(full_path)
  for i in tqdm(range(len(video))):
    if video[i] is not None:
      mmcv.imwrite(
        video[i],
        "{}/img_{:05d}.jpg".format(
          out_full_path,
          i + 1
        )
      )
    else:
      print(
        "😔 length inconsistent!"
        "early stop with {} out of {} frames".format(
          i + 1,
          len(video)
        )
      )
      break
  print("🤓 {} done with {} frames".format(video_name, len(video)))
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
  splits = glob.glob("{}/*list*.txt".format(ANNOTATIONS_ROOT))
  print(splits)
  for i in tqdm(range(len(splits))):
    file = open(splits[i], "r")
    temp = file.read()
    videos = temp.split("\n")
    frames_list = []
    for video in tqdm(range(len(videos))):
      frames = glob.glob(
        "{}/{}/*.jpg".format(
          FRAMES_ROOT,
          videos[video].split(".")[0]
        )
      )
      new_list.extend(frames)

    print(frames_list)


  '''classes = os.listdir(FRAMES_ROOT)
  for classname in classes:
    class_dir = path.join(FRAMES_ROOT, classname)
    fullpath_list = glob.glob(class_dir + "/*/*.avi")'''
  

if __name__ == "__main__":
  # get arguments from terminal
  parser = argparse.ArgumentParser(description = "prepare UCF101 dataset")
  parser.add_argument("--download", action="store_true")
  parser.add_argument("--decode_video", action="store_true")
  parser.add_argument("--build_list", action="store_true")
  args = parser.parse_args()

  if args.download:
    print("🤖 downloading UCF101 dataset")
    download_dataset()
  if args.decode_video:
    print("🤖 decoding videos to frames")
    decode_videos_to_frames()
  if args.build_list:
    print("🤖 generating training files")
    build_list()
