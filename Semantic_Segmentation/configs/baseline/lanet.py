from configs.base.gid import train, test, data, optimizer, learning_rate

config = dict(
    model=dict(
        type='LANet',
        params=dict(
            pretrained=True,
            in_channels=3,
            classes=6,
            loss=dict(
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
