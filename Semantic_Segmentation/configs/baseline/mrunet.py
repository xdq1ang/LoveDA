from configs.base.loveda import train, test, data, optimizer, learning_rate

config = dict(
    model=dict(
        type='mrUNet',
        params=dict(
            classes=7,
            num_channels=3,
            filters=32,
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
