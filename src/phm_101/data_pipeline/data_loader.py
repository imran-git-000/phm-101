from typing import TYPE_CHECKING

import numpy as np
import torch
from loguru import logger
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from phm_101.data_types.models import ChannelData

Sample = tuple[torch.Tensor, int, int]


class IMSDataset(Dataset[Sample]):
    """Overlapping windows cut inside each snapshot, never across snapshots.

    Returns (window, label, snapshot_index). The snapshot index is what
    aggregates window scores back to snapshot level at evaluation time.
    """

    def __init__(
        self,
        data: ChannelData,
        window_size: int,
        hop: int,
    ) -> None:
        # validation: window_size and hop must be positive, and window_size must not exceed the number of samples in a snapshot.
        if hop < 1:
            raise ValueError(f'hop must be >= 1, got {hop}')
        n_samples = data.signals.shape[1]
        if not 1 <= window_size <= n_samples:
            raise ValueError(
                f'window_size must be in [1, {n_samples}], got {window_size}'
            )

        # (n_snapshots, n_windows, window_size), a view: no data is copied
        self.windows = sliding_window_view(data.signals, window_size, axis=1)[
            :, ::hop
        ]
        self.labels = data.labels
        self.n_windows = self.windows.shape[1]

    def __len__(self) -> int:
        return self.windows.shape[0] * self.n_windows

    def __getitem__(self, index: int) -> Sample:
        snapshot, window = divmod(index, self.n_windows)
        # copy: the window is a strided view, which torch cannot wrap.
        # float32: the model's weights are float32, and signals reach us
        # as whatever dtype the transform produced.
        values = np.array(self.windows[snapshot, window], dtype=np.float32)
        return (
            torch.from_numpy(values).unsqueeze(0),
            int(self.labels[snapshot]),
            snapshot,
        )


class DataLoader:
    """Turn a ChannelData into a batched torch DataLoader of windows."""

    def __init__(
        self,
        window_size: int,
        hop: int,
        num_workers: int = 0,
        pin_memory: bool | None = None,
        seed: int | None = 0,
    ) -> None:
        self.logger = logger.bind(class_name=self.__class__.__name__)
        self.window_size = window_size
        self.hop = hop
        # num_workers > 0 pickles the dataset once per worker, which
        # materialises the window view. Keep it at 0 unless hop is large.
        self.num_workers = num_workers
        self.pin_memory = (
            torch.cuda.is_available() if pin_memory is None else pin_memory
        )
        self.seed = seed

    def get_dataloader(
        self,
        data: ChannelData,
        batch_size: int,
        train: bool,
    ) -> TorchDataLoader:
        """Windows are shuffled and the last partial batch dropped only for training."""
        dataset = IMSDataset(
            data=data, window_size=self.window_size, hop=self.hop
        )
        if train and len(dataset) < batch_size:
            raise ValueError(
                f'{len(dataset)} windows is fewer than batch_size '
                f'{batch_size}; with drop_last the loader would be empty'
            )

        self.logger.info(
            'Built {split} loader for {channel}: {n} windows',
            split='train' if train else 'eval',
            channel=data.channel,
            n=len(dataset),
        )
        generator = None
        if train and self.seed is not None:
            generator = torch.Generator().manual_seed(self.seed)
        return TorchDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            drop_last=train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            generator=generator,
        )
