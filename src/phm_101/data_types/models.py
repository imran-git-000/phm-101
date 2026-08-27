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


@dataclass
class ChannelSplit:
    """Chronological split of one channel. Train and val are healthy only."""

    train: ChannelData
    val: ChannelData
    test: ChannelData


@dataclass
class EvalResult:
    """Snapshot-level scores and the metrics derived from them."""

    scores: np.ndarray  # (n_snapshots,) anomaly score
    labels: np.ndarray  # (n_snapshots,) 0 healthy, 1 faulty
    threshold: float
    metrics: dict[str, float]


@dataclass
class TrainResult:
    """Training and validation loss curves."""

    train_losses: list[float]
    val_losses: list[float]
