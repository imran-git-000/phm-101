from phm_101.data_types.enums import ImsTests

CHANNELS: dict[ImsTests, list[str]] = {
    ImsTests.T1: [
        't1b1_x',
        't1b1_y',
        't1b2_x',
        't1b2_y',
        't1b3_x',
        't1b3_y',
        't1b4_x',
        't1b4_y',
    ],
    ImsTests.T2: ['t2b1', 't2b2', 't2b3', 't2b4'],
    ImsTests.T3: ['t3b1', 't3b2', 't3b3', 't3b4'],
}

N_SAMPLES = 20480
