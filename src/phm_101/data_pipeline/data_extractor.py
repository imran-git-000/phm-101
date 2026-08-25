from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from loguru import logger

from phm_101.data_types.models import ChannelData
from phm_101.utils.ims import N_SAMPLES, test_of


class DataExtractor:
    """Convert IMS raw text snapshots to parquet, and read them back.

    One parquet file per accelerometer, named ims_bearing_t1b3_x.
    """

    def __init__(self) -> None:
        self.logger = logger.bind(class_name=self.__class__.__name__)
        self.cache_dir = Path('data') / 'ims' / 'raw_signals'

        # expert judgment: first timestamp at which the channel is clearly
        # faulty (None = channel never fails)
        self.fault_onset: dict[str, str | None] = {
            't1b1_x': None,
            't1b1_y': None,
            't1b2_x': None,
            't1b2_y': None,
            't1b3_x': '2003-11-22 14:16:56',
            't1b3_y': '2003-11-22 14:16:56',
            't1b4_x': '2003-11-19 19:46:07',
            't1b4_y': '2003-11-19 19:46:07',
            't2b1': '2004-02-17 07:32:39',
            't2b2': None,
            't2b3': None,
            't2b4': None,
            't3b1': None,
            't3b2': None,
            't3b3': '2004-04-16 23:42:55',
            't3b4': None,
        }

        # last snapshot to keep
        self.valid_end: dict[str, str] = {
            '1st_test': '2003-11-25 23:39:56',
            '2nd_test': '2004-02-19 06:02:39',
            '3rd_test': '2004-04-18 02:32:55',
        }

    def load(self, channel: str) -> ChannelData:
        """Read one channel, drop shutdown snapshots and attach labels."""
        self.logger.info(
            'Loading IMS raw signals from parquet file for channel {channel}',
            channel=channel,
        )

        table = pq.read_table(
            self.cache_dir / f'ims_bearing_{channel}.parquet'
        )
        timestamps = table.column('timestamp').to_numpy(zero_copy_only=False)
        signals = (
            table.column('signal')
            .combine_chunks()
            .flatten()
            .to_numpy(zero_copy_only=False)
            .reshape(-1, N_SAMPLES)
        )

        # clean: keep only snapshots recorded before rig shutdown
        keep = timestamps <= np.datetime64(
            self.valid_end[test_of(channel).value]
        )
        timestamps, signals = timestamps[keep], signals[keep]

        # label: faulty from the onset timestamp onwards
        onset = self.fault_onset[channel]
        labels = (
            np.zeros(len(timestamps), dtype='int8')
            if onset is None
            else (timestamps >= np.datetime64(onset)).astype('int8')
        )
        self.logger.info(
            'Terminated loading IMS raw signals from parquet file for channel {channel}',
            channel=channel,
        )
        return ChannelData(
            channel=channel,
            timestamps=timestamps,
            signals=signals,
            labels=labels,
        )
