import os

DATASET_ROOT = "/dataset"

TRAIN_DATASET_ROOT = os.path.join(DATASET_ROOT, "train")
TEST_DATASET_ROOT = os.path.join(DATASET_ROOT, "test")
VALID_DATASET_ROOT = os.path.join(DATASET_ROOT, "valid")

TRAIN_FRAMES_ROOT = os.path.join(DATASET_ROOT, "train_frames")
TEST_FRAMES_ROOT = os.path.join(DATASET_ROOT, "test_frames")
VALID_FRAMES_ROOT = os.path.join(DATASET_ROOT, "valid_frames")

EPOCHS = 50
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32

MODEL_ROOT = "/model"