from configs.data.loveda.ToURBAN import SOURCE_DATA_CONFIG,TARGET_DATA_CONFIG, EVAL_DATA_CONFIG, TARGET_SET, source_dir
from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, Normalize, RandomCrop, RandomScale
from albumentations import OneOf, Compose
import ever as er
MODEL = 'ResNet'


IGNORE_LABEL = -1
MOMENTUM = 0.9
NUM_CLASSES = 7

SAVE_PRED_EVERY = 1000
SNAPSHOT_DIR = './log/train_in_loveda/pycda'

#Hyper Paramters
WEIGHT_DECAY = 0.0005
LEARNING_RATE = 1e-3
NUM_STEPS = 15000
NUM_STEPS_STOP = 10000  # Use damping instead of early stopping
PREHEAT_STEPS = int(NUM_STEPS / 20)
POWER = 0.9
EVAL_EVERY=1000

# Loss
LAMBDA_SEG = 1.0
LAMBDA_TRADE_OFF=1
CONF_THRESHOLD=0.5
BOX_SIZE = [2, 4, 8]
MERGE_1X1 = True
LAMBDA_BALANCE = 1
LAMBDA_PSEUDO = 0.5



TARGET_SET = TARGET_SET
SOURCE_DATA_CONFIG = SOURCE_DATA_CONFIG
TARGET_DATA_CONFIG=TARGET_DATA_CONFIG