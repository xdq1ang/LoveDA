from configs.URBAN import SOURCE_DATA_CONFIG, EVAL_DATA_CONFIG, source_dir
from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, Normalize, RandomCrop, RandomScale
from albumentations import OneOf, Compose
import ever as er
MODEL = 'ResNet'


IGNORE_LABEL = -1
MOMENTUM = 0.9
NUM_CLASSES = 7

SAVE_PRED_EVERY = 2000
SNAPSHOT_DIR = './log/baseline/2urban'

#Hyper Paramters
WEIGHT_DECAY = 0.0005
LEARNING_RATE = 1e-3
NUM_STEPS = 25000
NUM_STEPS_STOP = 20000  # Use damping instead of early stopping
PREHEAT_STEPS = 0
POWER = 0.9
EVAL_EVERY=700
TARGET_SET='URABN'
SOURCE_DATA_CONFIG = dict(
    image_dir=source_dir['image_dir'],
    mask_dir=source_dir['mask_dir'],
    transforms=Compose([
        RandomCrop(512, 512),
        OneOf([
            HorizontalFlip(True),
            VerticalFlip(True),
            RandomRotate90(True)
        ], p=0.75),
        Normalize(mean=(123.675, 116.28, 103.53),
                  std=(58.395, 57.12, 57.375),
                  max_pixel_value=1, always_apply=True),
        er.preprocess.albu.ToTensor()

    ]),
    CV=dict(k=10, i=-1),
    training=True,
    batch_size=12,
    num_workers=2,
    drop_last=True
)
EVAL_DATA_CONFIG=EVAL_DATA_CONFIG