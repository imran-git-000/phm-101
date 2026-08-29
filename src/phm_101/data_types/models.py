from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from torch import Tensor

    from phm_101.config.configs import DataConfig, ModelConfig
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
class Metrics:
    """Detection quality at one threshold. tpr is recall under another name."""

    accuracy: float
    precision: float
    recall: float
    fpr: float
    tpr: float
    auroc: float  # nan when the channel never fails


@dataclass
class EvalResult:
    """Snapshot-level scores and the metrics derived from them."""

    scores: np.ndarray  # (n_snapshots,) anomaly score
    labels: np.ndarray  # (n_snapshots,) 0 healthy, 1 faulty
    threshold: float
    metrics: Metrics


@dataclass
class TrainResult:
    """Training and validation loss curves."""

    train_losses: list[float]
    val_losses: list[float]


@dataclass
class Checkpoint:
    """A trained model plus everything needed to score with it again."""

    state_dict: dict[str, Tensor]
    model_config: ModelConfig
    data_config: DataConfig
    channel: ImsChannel


@dataclass
class RunResult:
    """One channel taken end to end, and where its artifacts landed."""

    channel: ImsChannel
    eval_result: EvalResult
    train_result: TrainResult | None  # None when a checkpoint was loaded
