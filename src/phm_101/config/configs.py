from dataclasses import dataclass
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class AutoencoderConfig:
    """Every tunable knob of the model lives here."""

    window_size: int
    in_channels: int
    # one downsampling block per entry; length sets the depth
    channels: tuple[int, ...]
    kernel_size: int
    latent_dim: int
    dropout: float
    batch_norm: bool

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

    epochs: int
    learning_rate: float
    weight_decay: float
    patience: int  # early stopping; 0 disables it
    device: str
    seed: int | None  # training-time randomness; None disables seeding


@dataclass
class Config:
    """The whole config file: one field per top-level YAML section."""

    model: AutoencoderConfig
    training: TrainConfig


def load_config(path: Path) -> Config:
    """Read the YAML file into the config dataclasses."""
    raw = yaml.safe_load(path.read_text(encoding='utf-8'))
    model = dict(raw['model'])
    # YAML has no tuples, and the model concatenates channels onto one
    model['channels'] = tuple(model['channels'])
    return Config(
        model=AutoencoderConfig(**model),
        training=TrainConfig(**raw['training']),
    )
