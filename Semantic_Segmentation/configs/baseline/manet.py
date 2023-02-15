from configs.base.gid import train, test, data, optimizer, learning_rate

config = dict(
    model=dict(
        type='MANet',
        params=dict(
            encoder_name='resnet50',
            classes=6,
            encoder_weights='imagenet',
            loss=dict(
                ignore_index=-1,
                ce=dict()
            )
        )
    ),
    data=data,
    optimizer=optimizer,
    learning_rate=learning_rate,
    train=train,
    test=test
)
