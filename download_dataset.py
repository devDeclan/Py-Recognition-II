import os
import argparse
from lib.config import DATASET_ROOT

def download_dataset():
  # checking if dataset root exist otherwise create it
  if not os.path.exists(DATASET_ROOT):
    os.makedirs(DATASET_ROOT)

  # check if the dataset file has already been downloaded
  # otherwise download it
  # (ano get data like that😂)
  dataset_rar = os.path.join(DATASET_ROOT, "dataset.rar")
  if not os.path.exists(dataset_rar):
    print("👾 downloading dataset file")
    dataset_url = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
    command = " ".join([
      "wget", "--quiet",
      "--directory-prefix", DATASET_ROOT,
      "--output-document", dataset_rar,
      dataset_url
    ])
    os.system(command)
    print("👾 dataset file downloaded")
  else:
    print("👾 dataset file already exists, skipping download")

  # extract the videos
  print("👾 extracting dataset file")
  videos_path = os.path.join(DATASET_ROOT, "videos")
  command = " ".join([
    "unrar", "x",
    dataset_rar, DATASET_ROOT,
    ">", "/dev/null",
    "&&", "mv",
    os.path.join(DATASET_ROOT, "UCF-101"),
    videos_path
  ])
  os.system(command)
  print("👾 dataset file downloaded and extracted")

  # check if the annotations file has already been download
  # otherwise download it
  annotations_zip = os.path.join(DATASET_ROOT, "annotations.zip")
  if not os.path.exists(annotations_zip):
    print("👾 downloading annotations file")
    annotations_url = "https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip"
    command = " ".join([
      "wget", "--quiet",
      "--directory-prefix", DATASET_ROOT,
      "--output-document", annotations_zip,
      annotations_url
    ])
    os.system(command)
    print("👾 annotations file downloaded")
  else:
    print("👾 annotations file already exists, skipping download")

  # extract the annotations
  print("👾 extracting annotations file")
  annotations_path = os.path.join(DATASET_ROOT, "annotations")
  command = " ".join([
    "unzip",
    "-q", annotations_zip,
    "-d", DATASET_ROOT,
    "&&", "mv",
    os.path.join(DATASET_ROOT, "ucfTrainTestlist"),
    annotations_path
  ])
  os.system(command)
  print("👾 annotations file downloaded and extracted")

def decode_videos_to_frames():
  pass
  # checking if frames root exist otherwise create it
  #if not os.path.exists(DATASET_ROOT):
  # os.makedirs(DATASET_ROOT)

def build_file_list():
  pass

if __name__ == "__main__":
  # get arguments from terminal
  parser = argparse.ArgumentParser(description = "prepare UCF101 dataset")
  parser.add_argument("--download", action="store_true", default=True)
  parser.add_argument("--decode_video", action="store_true", default=True)
  parser.add_argument("--build_file_list", action="store_true", default=True)
  args = parser.parse_args()

  if args.download:
    print("🤖 downloading UCF101 dataset")
    download_dataset()
  if args.decode_video:
    print("🤖 decoding videos to frames")
    decode_videos_to_frames()
  if args.build_file_list:
    print("🤖 generating training files")
    build_file_list()
