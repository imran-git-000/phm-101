from typing import TYPE_CHECKING

import torch
from loguru import logger
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from phm_101.data_types.models import ChannelData


class IMSDataset(Dataset):
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
        # (n_snapshots, n_windows, window_size), a view: no data is copied
        self.windows = sliding_window_view(data.signals, window_size, axis=1)[
            :, ::hop
        ]
        self.labels = data.labels
        self.n_windows = self.windows.shape[1]

    def __len__(self) -> int:
        return self.windows.shape[0] * self.n_windows

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        snapshot, window = divmod(index, self.n_windows)
        return (
            torch.from_numpy(self.windows[snapshot, window].copy()),
            int(self.labels[snapshot]),
            snapshot,
        )


class DataLoader:
    """Turn a ChannelData into a batched torch DataLoader of windows."""

    def __init__(
        self,
        window_size: int,
        hop: int,
        batch_size: int,
    ) -> None:
        self.logger = logger.bind(class_name=self.__class__.__name__)
        self.window_size = window_size
        self.hop = hop
        self.batch_size = batch_size

    def get_dataloader(
        self,
        data: ChannelData,
        train: bool,
    ) -> TorchDataLoader:
        """Windows are shuffled and the last partial batch dropped only for training."""
        dataset = IMSDataset(
            data=data, window_size=self.window_size, hop=self.hop
        )
        self.logger.info(
            'Built {split} loader for {channel}: {n} windows',
            split='train' if train else 'eval',
            channel=data.channel,
            n=len(dataset),
        )
        return TorchDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train,
            drop_last=train,
        )
