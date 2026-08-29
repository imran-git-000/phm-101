from phm_101.data_types.enums import ImsChannel, ImsTest

CHANNELS: dict[ImsTest, list[ImsChannel]] = {
    ImsTest.T1: [
        ImsChannel.T1B1x,
        ImsChannel.T1B1y,
        ImsChannel.T1B2x,
        ImsChannel.T1B2y,
        ImsChannel.T1B3x,
        ImsChannel.T1B3y,
        ImsChannel.T1B4x,
        ImsChannel.T1B4y,
    ],
    ImsTest.T2: [
        ImsChannel.T2B1,
        ImsChannel.T2B2,
        ImsChannel.T2B3,
        ImsChannel.T2B4,
    ],
    ImsTest.T3: [
        ImsChannel.T3B1,
        ImsChannel.T3B2,
        ImsChannel.T2B3,
        ImsChannel.T2B4,
    ],
}

N_SAMPLES = 20480


def test_of(channel: ImsChannel) -> ImsTest:
    """The test a channel belongs to."""
    for test, channels in CHANNELS.items():
        if channel in channels:
            return test
    raise ValueError(f'unknown channel: {channel}')
