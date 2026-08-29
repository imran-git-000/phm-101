from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import yaml

from phm_101.data_types.enums import Aggregation, ImsChannels, ModelName

if TYPE_CHECKING:
    from pathlib import Path


# kw_only so a subclass can add a required field under a defaulted base one;
# without it dataclass inheritance forces every later field to have a default
@dataclass(kw_only=True)
class ModelConfig:
    """What every model needs, whatever family it belongs to."""

    name: str


@dataclass(kw_only=True)
class Conv1DAutoencoderConfig(ModelConfig):
    """Hyperparameters of the 1D convolutional autoencoder."""

    name: str = ModelName.CONV1D_AUTOENCODER.value  # default for this subclass
    # one downsampling block per entry; length sets the depth
    channels: tuple[int, ...] = (16, 32, 64, 128)
    kernel_size: int = (
        7  # must be odd, so padding = kernel_size // 2 halves exactly
    )
    latent_dim: int = 32
    dropout: float = 0.0
    batch_norm: bool = True

    @property
    def n_blocks(self) -> int:
        return len(self.channels)


@dataclass
class TrainConfig:
    """Training loop configuration."""

    epochs: int
    learning_rate: float
    weight_decay: float
    patience: int  # early stopping; 0 disables it
    device: str
    seed: int | None  # training-time randomness; None disables seeding


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""

    channel: ImsChannels
    window_size: int
    hop: int  # stride between windows; hop == window_size means no overlap
    in_channels: int
    batch_size: int
    num_workers: int


@dataclass
class EvalConfig:
    """Every tunable knob of the evaluation lives here."""

    aggregation: Aggregation
    quantile: float  # threshold percentile on healthy validation scores


@dataclass
class Config:
    """The whole config file: one field per top-level YAML section."""

    model: ModelConfig
    training: TrainConfig
    data: DataConfig
    evaluation: EvalConfig


class ConfigLoader:
    """Class containing static methods to load a config file into the dataclasses."""

    @staticmethod
    def build_model_config(raw_dict: dict[str, Any]) -> ModelConfig:
        """Pick the subclass whose default name matches the YAML."""
        for config_class in ModelConfig.__subclasses__():
            if (
                config_class.__dataclass_fields__['name'].default
                == raw_dict['name']
            ):
                return config_class(**raw_dict)
        raise ValueError(f'unknown model {raw_dict["name"]!r}')

    @staticmethod
    def load_config(path: Path) -> Config:
        """Read the YAML file into the config dataclasses."""
        raw_dict = yaml.safe_load(path.read_text(encoding='utf-8'))
        return Config(
            model=ConfigLoader.build_model_config(raw_dict=raw_dict['model']),
            training=TrainConfig(**raw_dict['training']),
            data=DataConfig(**raw_dict['data']),
            evaluation=EvalConfig(**raw_dict['evaluation']),
        )
