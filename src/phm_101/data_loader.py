import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
from tqdm import tqdm

from phm_101.data_types.models import ChannelData


class DataLoader:
    """Convert IMS raw text snapshots to parquet, and read them back.

    One parquet file per accelerometer, named ims_bearing_t1b3_x.
    """

    def __init__(self) -> None:
        load_dotenv()
        data_root = os.environ.get('DATA_ROOT')
        if not data_root:
            raise ValueError('DATA_ROOT env var is not set (see .env.example)')
        self.raw_dir = Path(data_root) / 'time_series' / 'bearing' / 'ims'
        self.cache_dir = Path('data') / 'ims' / 'raw_signals'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.channels: dict[str, list[str]] = {
            '1st_test': [
                't1b1_x',
                't1b1_y',
                't1b2_x',
                't1b2_y',
                't1b3_x',
                't1b3_y',
                't1b4_x',
                't1b4_y',
            ],
            '2nd_test': ['t2b1', 't2b2', 't2b3', 't2b4'],
            '3rd_test': ['t3b1', 't3b2', 't3b3', 't3b4'],
        }

        self.n_samples = 20480
        self.schema = pa.schema(
            [
                ('timestamp', pa.timestamp('s')),
                ('signal', pa.list_(pa.float32(), self.n_samples)),
            ]
        )

        # expert judgment: first timestamp at which the channel is clearly
        # faulty (None = channel never fails)
        self.fault_onset: dict[str, str | None] = {
            't1b1_x': None,
            't1b1_y': None,
            't1b2_x': None,
            't1b2_y': None,
            't1b3_x': '2003-11-21 00:32:00',
            't1b3_y': '2003-11-21 00:32:00',
            't1b4_x': '2003-11-21 00:32:00',
            't1b4_y': '2003-11-21 00:32:00',
            't2b1': '2004-02-17 12:32:00',
            't2b2': None,
            't2b3': None,
            't2b4': None,
            't3b1': None,
            't3b2': None,
            't3b3': '2004-04-15 12:32:00',
            't3b4': None,
        }

        # last usable snapshot: later ones are recorded during rig shutdown
        self.valid_end: dict[str, str] = {
            '1st_test': '2003-11-24 18:22:00',
            '2nd_test': '2004-02-19 00:42:00',
            '3rd_test': '2004-04-18 00:42:00',
        }

    def save(self) -> None:
        """Convert every test to one parquet file per channel."""
        # iterate over test directories
        for test_dir_name, channels in self.channels.items():
            # instantiate one ParquetWriter and a buffer for each channel
            writers: dict[str, pq.ParquetWriter] = {
                channel: pq.ParquetWriter(
                    self.cache_dir / f'ims_bearing_{channel}.parquet',
                    self.schema,
                    compression='zstd',
                )
                for channel in channels
            }
            try:
                # iterate over snapshots of the current test directory
                for snapshot_path in tqdm(
                    self._snapshots(test_dir_name=test_dir_name),
                    desc=test_dir_name,
                ):
                    # extract snapshot as dataframe
                    df = pd.read_csv(
                        snapshot_path,
                        header=None,
                        sep='\t',
                        names=channels,
                        dtype='float32',
                    )
                    timestamp = self._timestamp(path=snapshot_path)
                    # for each channel in current snapshot
                    for channel in channels:
                        # extract signal and write to parquet
                        signal = df[channel].to_numpy()
                        writers[channel].write_table(
                            pa.table(
                                {
                                    'timestamp': pa.array(
                                        [timestamp], pa.timestamp('s')
                                    ),
                                    'signal': pa.array(
                                        [signal],
                                        pa.list_(pa.float32(), self.n_samples),
                                    ),
                                },
                                schema=self.schema,
                            )
                        )
            finally:
                # close all writers to flush buffers
                for writer in writers.values():
                    writer.close()

    def load(self, channel: str) -> ChannelData:
        """Read one channel, drop shutdown snapshots and attach labels."""
        table = pq.read_table(
            self.cache_dir / f'ims_bearing_{channel}.parquet'
        )
        timestamps = table.column('timestamp').to_numpy(zero_copy_only=False)
        signals = (
            table.column('signal')
            .combine_chunks()
            .flatten()
            .to_numpy(zero_copy_only=False)
            .reshape(-1, self.n_samples)
        )

        # clean: keep only snapshots recorded before rig shutdown
        test_dir_name = next(
            test_dir
            for test_dir, channels in self.channels.items()
            if channel in channels
        )
        keep = timestamps <= np.datetime64(self.valid_end[test_dir_name])
        timestamps, signals = timestamps[keep], signals[keep]

        # label: faulty from the onset timestamp onwards
        onset = self.fault_onset[channel]
        labels = (
            np.zeros(len(timestamps), dtype='int8')
            if onset is None
            else (timestamps >= np.datetime64(onset)).astype('int8')
        )

        return ChannelData(
            channel=channel,
            timestamps=timestamps,
            signals=signals,
            labels=labels,
        )

    def _snapshots(self, test_dir_name: str) -> list[Path]:
        """All snapshot files of a test, sorted chronologically."""
        snapshot_paths = [
            snapshot_path
            for snapshot_path in (self.raw_dir / test_dir_name).iterdir()
            if snapshot_path.is_file()
        ]
        return sorted(snapshot_paths, key=self._timestamp)

    @staticmethod
    def _timestamp(path: Path) -> datetime:
        return datetime.strptime(path.name, '%Y.%m.%d.%H.%M.%S')
