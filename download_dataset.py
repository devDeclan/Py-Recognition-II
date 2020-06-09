import os
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

	# check if the annotations file has already been download
	# otherwise download it
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

def decode_videos_to_frames():
	pass