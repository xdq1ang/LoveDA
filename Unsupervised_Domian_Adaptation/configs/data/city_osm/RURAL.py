from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, Normalize, RandomCrop, RandomScale, Resize
from albumentations import OneOf, Compose
import ever as er


TARGET_SET = 'RURAL'

source_dir = dict(
    image_dir=[
        r"D:\UDA_Datasets\CITY_OSM\train\chicago\img"
    ],
    mask_dir=[
        r"D:\UDA_Datasets\CITY_OSM\train\chicago\lab"
    ],
    val_image_dir=[
        r"D:\UDA_Datasets\CITY_OSM\val\chicago\img"
    ],
    val_mask_dir=[
        r"D:\UDA_Datasets\CITY_OSM\val\chicago\lab"
    ],
)


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
    batch_size=6,
    num_workers=2,
    drop_last=True
)



EVAL_DATA_CONFIG = dict(
    image_dir=source_dir['val_image_dir'],
    mask_dir=source_dir['val_mask_dir'],
    transforms=Compose([
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
