from configs.base.isprs import train, test, data, optimizer, learning_rate
import torch.nn as nn
config = dict(
    model=dict(
        type='PSPNet_DensePPM',
        params=dict(
            encoder=dict(
                resnet_type='resnet50',
                include_conv5=True,
                batchnorm_trainable=True,
                pretrained=True,
                freeze_at=0,
                # 16 or 32
                output_stride=16,
                with_cp=(False, False, False, False),
                norm_layer=nn.BatchNorm2d,
            ),
            classes=6,
            resume_from_last=True,
            loss=dict(
                ignore_index=-1,
                ce=dict()
            ),
            psp=dict(
                sizes=(2, 3, 4, 5),
                psp_size=1024,
                deep_features_size=1024
            )
        )
    ),
    data=data,
    optimizer=optimizer,
    learning_rate=learning_rate,
    train=train,
    test=test
)