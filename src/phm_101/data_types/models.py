from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from torch import Tensor

    from phm_101.config.configs import ModelConfig
    from phm_101.data_types.enums import ImsChannel


@dataclass
class ChannelData:
    """One accelerometer channel, cleaned and labelled."""

    channel: ImsChannel
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


@dataclass
class Checkpoint:
    """A trained model plus everything needed to score with it again.

    The z-score statistics belong here: scoring with statistics other than
    the ones the model trained under silently shifts every reconstruction
    error, so the weights alone are not enough to reproduce a run.
    """

    state_dict: dict[str, Tensor]
    config: ModelConfig
    mean: float
    std: float
    channel: ImsChannel


@dataclass
class RunResult:
    """One channel taken end to end, and where its artifacts landed."""

    channel: ImsChannel
    eval_result: EvalResult
    output_dir: Path
    train_result: TrainResult | None  # None when a checkpoint was loaded
