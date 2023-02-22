from albumentations import Compose, OneOf, Normalize
from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, RandomCrop, Resize
import ever as er
import cv2

data = dict(
    train=dict(
        type='CITY_OSMLoader',
        params=dict(
            image_dir=[
                r'D:\UDA_Datasets\CITY_OSM\train\chicago\img',
                r'D:\UDA_Datasets\CITY_OSM\train\zurich\img',
            ],
            mask_dir=[
                r'D:\UDA_Datasets\CITY_OSM\train\chicago\lab',
                r'D:\UDA_Datasets\CITY_OSM\train\zurich\lab',
            ],
            
            transforms=Compose([
                RandomCrop(512, 512),
                # Resize(256, 256, cv2.INTER_NEAREST, True),
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
            batch_size=8,
            num_workers=2,
        ),
    ),
    test=dict(
        type='CITY_OSMLoader',
        params=dict(
            image_dir=[
                r'D:\UDA_Datasets\CITY_OSM\val\chicago\img',
                r'D:\UDA_Datasets\CITY_OSM\val\zurich\img',
            ],
            mask_dir=[
                r'D:\UDA_Datasets\CITY_OSM\val\chicago\lab',
                r'D:\UDA_Datasets\CITY_OSM\val\zurich\lab',
            ],
            transforms=Compose([
                # Resize(512, 512, cv2.INTER_NEAREST, True),
                Normalize(mean=(123.675, 116.28, 103.53),
                          std=(58.395, 57.12, 57.375),
                          max_pixel_value=1, always_apply=True),
                er.preprocess.albu.ToTensor()

            ]),
            CV=dict(k=10, i=-1),
            training=False,
            batch_size=1,
            num_workers=0,
        ),
    ),
)
optimizer = dict(
    type='sgd',
    params=dict(
        momentum=0.9,
        weight_decay=0.0001
    ),
    grad_clip=dict(
        max_norm=35,
        norm_type=2,
    )
)
learning_rate = dict(
    type='poly',
    params=dict(
        base_lr=0.001,
        power=0.9,
        max_iters=15000,
    ))
train = dict(
    forward_times=1,
    num_iters=15000,
    eval_per_epoch=True,
    summary_grads=False,
    summary_weights=False,
    distributed=True,
    apex_sync_bn=True,
    sync_bn=True,
    eval_after_train=True,
    log_interval_step=50,
    save_ckpt_interval_epoch=1000,
    eval_interval_epoch=5,
    resume_from_last=True
)

test = dict(

)
