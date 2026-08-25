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
class AutoencoderConfig:
    """Every tunable knob of the model lives here."""

    window_size: int = 1024
    in_channels: int = 1
    # one downsampling block per entry; length sets the depth
    channels: tuple[int, ...] = (16, 32, 64, 128)
    kernel_size: int = 7
    latent_dim: int = 32
    dropout: float = 0.0
    batch_norm: bool = True

    @property
    def n_blocks(self) -> int:
        return len(self.channels)

    @property
    def bottleneck_length(self) -> int:
        """Sequence length after all stride-2 blocks."""
        return self.window_size // 2**self.n_blocks


@dataclass
class TrainConfig:
    """Every tunable knob of the training loop lives here."""

    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 10  # early stopping; 0 disables it
    grad_clip: float = 0.0  # max grad norm; 0 disables it
    device: str = 'auto'


@dataclass
class EvalResult:
    """Snapshot-level scores and the metrics derived from them."""

    scores: np.ndarray  # (n_snapshots,) anomaly score
    labels: np.ndarray  # (n_snapshots,) 0 healthy, 1 faulty
    threshold: float
    metrics: dict[str, float]
