from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass
class ChannelData:
    """One accelerometer channel, cleaned and labelled."""

    channel: str
    timestamps: np.ndarray  # (n_snapshots,) datetime64[s]
    signals: np.ndarray  # (n_snapshots, 20480) float32
    labels: np.ndarray  # (n_snapshots,) int8, 0 = healthy, 1 = faulty

    def __len__(self) -> int:
        return len(self.timestamps)
