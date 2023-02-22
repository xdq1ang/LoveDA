from configs.base.city_osm import train, test, data, optimizer, learning_rate

config = dict(
    model=dict(
        type='CENet',
        params=dict(
            classes=3,
            num_channels=3,
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
