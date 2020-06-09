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
	print("👾 downloading dataset file")
	dataset_rar = os.path.join(DATASET_ROOT, "dataset.rar")
	if not os.path.exists(dataset_rar):
		dataset_url = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
		command = " ".join([
			"wget",
			"--directory-prefix", DATASET_ROOT,
			"--output-document", "dataset.rar",
			dataset_url
		])
		os.system(command)

		# extract the videos
		videos_path = os.path.join(DATASET_ROOT, "videos")
		command = " ".join([
			"unrar",
			"e", dataset_rar,
			videos_path
		])
		os.system(command)
		print("👾 file downloaded and extracted")
	else:
		print("👾 file already exists, skipping")

	# check if the annotations file has already been download
	# otherwise download it
	print("👾 downloading annotations file")
	annotations_zip = os.path.join(DATASET_ROOT, "annotations.zip")
	if not os.path.exists(annotations_zip):
		annotations_url = "https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip"
		command = " ".join([
			"wget",
			"--directory-prefix", DATASET_ROOT,
			"--output-document", "annotations.zip",
			dataset_url
		])
		os.system(command)

		# extract the annotations
		annotations_path = os.path.join(DATASET_ROOT, "annotations")
		command = " ".join([
			"unzip",
			"-q", annotations_zip,
			"-d", annotations_path
		])
		os.system(command)
		print("👾 file downloaded and extracted")
	else:
		print("👾 file already exists, skipping")

def decode_videos_to_frames():
	pass
	# checking if frames root exist otherwise create it
	#if not os.path.exists(DATASET_ROOT):
	#	os.makedirs(DATASET_ROOT)

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
