from configs.base.loveda import train, test, data, optimizer, learning_rate

config = dict(
    model=dict(
        type='ResUNetA',
        params=dict(
            classes=7,
            in_channel=3,
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