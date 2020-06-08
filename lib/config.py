import os

DATASET_ROOT = "/dataset"
TRAIN_DATASET_ROOT = os.path.join(DATASET_ROOT, "train")
TEST_DATASET_ROOT = os.path.join(DATASET_ROOT, "test")
VALID_DATASET_ROOT = os.path.join(DATASET_ROOT, "valid")

EPOCHS = 50
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32

MODEL_ROOT = "/model"