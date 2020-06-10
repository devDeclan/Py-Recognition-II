import os

DATASET_ROOT = "dataset"
FRAMES_ROOT = os.path.join(DATASET_ROOT, "frames")
VIDEOS_ROOT = os.path.join(DATASET_ROOT, "videos")
ANNOTATIONS_ROOT = os.path.join(DATASET_ROOT, "annotations")

EPOCHS = 50
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 2
WORKERS = 8

MODEL_ROOT = "model"