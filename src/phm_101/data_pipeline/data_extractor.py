from typing import TYPE_CHECKING

import numpy as np
import pyarrow.parquet as pq
from loguru import logger

from phm_101.data_types.enums import ImsChannel, ImsTest
from phm_101.data_types.models import ChannelData
from phm_101.utils.ims import N_SAMPLES, test_of

if TYPE_CHECKING:
    from pathlib import Path


class DataExtractor:
    """Convert IMS channel parquet data into ChannelData objects, dropping shutdown snapshots and attaching labels."""

    def __init__(self, raw_signals_dir: Path) -> None:
        self.logger = logger.bind(class_name=self.__class__.__name__)
        self.cache_dir = raw_signals_dir

        # expert judgment: first timestamp at which the channel is clearly
        # faulty (None = channel never fails)
        self.fault_onset: dict[ImsChannel, str | None] = {
            ImsChannel.T1B1x: None,
            ImsChannel.T1B1y: None,
            ImsChannel.T1B2x: None,
            ImsChannel.T1B2y: None,
            ImsChannel.T1B3x: '2003-11-22 14:16:56',
            ImsChannel.T1B3y: '2003-11-22 14:16:56',
            ImsChannel.T1B4x: '2003-11-19 19:46:07',
            ImsChannel.T1B4y: '2003-11-19 19:46:07',
            ImsChannel.T2B1: '2004-02-17 07:32:39',
            ImsChannel.T2B2: None,
            ImsChannel.T2B3: None,
            ImsChannel.T2B4: None,
            ImsChannel.T3B1: None,
            ImsChannel.T3B2: None,
            ImsChannel.T3B3: '2004-04-16 23:42:55',
            ImsChannel.T3B4: None,
        }

        # last snapshot to keep
        self.valid_end: dict[ImsTest, str] = {
            ImsTest.T1: '2003-11-25 23:39:56',
            ImsTest.T2: '2004-02-19 06:02:39',
            ImsTest.T3: '2004-04-18 02:32:55',
        }

    def load(self, channel: ImsChannel) -> ChannelData:
        """Read one channel, drop shutdown snapshots and attach labels."""
        self.logger.info(
            'Loading IMS raw signals from parquet file for channel {channel}',
            channel=channel,
        )

        table = pq.read_table(
            self.cache_dir / f'ims_bearing_{channel.value}.parquet'
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
        keep = timestamps <= np.datetime64(self.valid_end[test_of(channel)])
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
