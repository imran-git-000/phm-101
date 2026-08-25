import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

from phm_101.utils.ims import CHANNELS, N_SAMPLES

if TYPE_CHECKING:
    from phm_101.data_types.enums import ImsTests


class DataExtractor:
    def __init__(self) -> None:
        self.logger = logger.bind(class_name=self.__class__.__name__)
        load_dotenv()
        data_root = os.environ.get('DATA_ROOT')
        if not data_root:
            raise ValueError('DATA_ROOT env var is not set (see .env.example)')
        self.raw_dir = Path(data_root) / 'time_series' / 'bearing' / 'ims'
        self.cache_dir = Path('data') / 'ims' / 'raw_signals'

        self.schema = pa.schema(
            [
                ('timestamp', pa.timestamp('s')),
                ('signal', pa.list_(pa.float32(), N_SAMPLES)),
            ]
        )

    def save(self) -> None:
        """Convert every test to one parquet file per channel."""
        self.logger.info(
            'Saving IMS raw signals to parquet files in {dir}',
            dir=self.cache_dir,
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # iterate over test directories
        for test_dir_name, channels in CHANNELS.items():
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
                    desc=test_dir_name.value,
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
                                        pa.list_(pa.float32(), N_SAMPLES),
                                    ),
                                },
                                schema=self.schema,
                            )
                        )
            finally:
                # close all writers to flush buffers
                for writer in writers.values():
                    writer.close()
        self.logger.info(
            'Saved IMS raw signals to parquet files in {dir}',
            dir=self.cache_dir,
        )

    def _snapshots(self, test_dir_name: ImsTests) -> list[Path]:
        """All snapshot files of a test, sorted chronologically."""
        snapshot_paths = [
            snapshot_path
            for snapshot_path in (self.raw_dir / test_dir_name.value).iterdir()
            if snapshot_path.is_file()
        ]
        return sorted(snapshot_paths, key=self._timestamp)

    @staticmethod
    def _timestamp(path: Path) -> datetime:
        return datetime.strptime(path.name, '%Y.%m.%d.%H.%M.%S')


def main() -> None:
    logger.info('Starting IMS raw signals to parquet conversion')
    DataExtractor().save()


if __name__ == '__main__':
    main()
