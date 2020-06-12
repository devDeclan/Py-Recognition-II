import os
from os import path
from config import DATASET_ROOT, VIDEOS_ROOT, ANNOTATIONS_ROOT

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

if __name__ == "__main__":
  print("🤖 downloading UCF101 dataset")
  download_dataset()