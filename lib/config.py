import os

DATASET_ROOT = "dataset"
FRAMES_ROOT = os.path.join(DATASET_ROOT, "frames")
VIDEOS_ROOT = os.path.join(DATASET_ROOT, "videos")
ANNOTATIONS_ROOT = os.path.join(DATASET_ROOT, "annotations")

EPOCHS = 10
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
WORKERS = 8

MODEL_ROOT = "model"