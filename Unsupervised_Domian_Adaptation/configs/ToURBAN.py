from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, Normalize, RandomCrop, RandomScale, Resize
from albumentations import OneOf, Compose
import ever as er


TARGET_SET = 'URBAN'

source_dir = dict(
    image_dir=[
        './LoveDA/Train/Rural/images_png/',
    ],
    mask_dir=[
        './LoveDA/Train/Rural/masks_png/',
    ],
)
target_dir = dict(
    image_dir=[
        './LoveDA/Val/Urban/images_png/',
    ],
    mask_dir=[
        './LoveDA/Val/Urban/masks_png/',
    ]
)

test_dir = dict(
    image_dir=[
        './LoveDA/Test/Urban/images_png/'
    ]
)


SOURCE_DATA_CONFIG = dict(
    image_dir=source_dir['image_dir'],
    mask_dir=source_dir['mask_dir'],
    transforms=Compose([
        RandomCrop(512, 512),
        # Resize(512,512),
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
    batch_size=6,
    num_workers=2,
    drop_last=True
)


TARGET_DATA_CONFIG = dict(
    image_dir=target_dir['image_dir'],
    mask_dir=target_dir['mask_dir'],
    transforms=Compose([
        RandomCrop(512, 512),
        # Resize(512,512),
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
    batch_size=6,
    num_workers=2,
    drop_last=True
)

EVAL_DATA_CONFIG = dict(
    image_dir=target_dir['image_dir'],
    mask_dir=target_dir['mask_dir'],
    transforms=Compose([
        # Resize(512, 512),
        Normalize(mean=(123.675, 116.28, 103.53),
                  std=(58.395, 57.12, 57.375),
                  max_pixel_value=1, always_apply=True),
        er.preprocess.albu.ToTensor()

    ]),
    CV=dict(k=10, i=-1),
    training=False,
    batch_size=4,
    num_workers=0,
    drop_last=False
)

TEST_DATA_CONFIG = dict(
    image_dir=test_dir['image_dir'],
    mask_dir=test_dir['image_dir'],
    transforms=Compose([
        # Resize(512, 512),
        Normalize(mean=(123.675, 116.28, 103.53),
                  std=(58.395, 57.12, 57.375),
                  max_pixel_value=1, always_apply=True),
        er.preprocess.albu.ToTensor()

    ]),
    CV=dict(k=10, i=-1),
    training=False,
    batch_size=8,
    num_workers=6,
    drop_last=False
)