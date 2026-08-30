from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import yaml

from phm_101.data_types.enums import (
    Aggregation,
    ImsChannel,
    ModelName,
    Paradigm,
)

if TYPE_CHECKING:
    from pathlib import Path


# kw_only so a subclass can add a required field under a defaulted base one;
# without it dataclass inheritance forces every later field to have a default
@dataclass(kw_only=True)
class ModelConfig:
    """What every model needs, whatever family it belongs to."""

    name: ModelName
    paradigm: Paradigm

    def __post_init__(self) -> None:
        self.name = ModelName(self.name)
        self.paradigm = Paradigm(self.paradigm)


@dataclass(kw_only=True)
class Conv1DAutoencoderConfig(ModelConfig):
    """Hyperparameters of the 1D convolutional autoencoder."""

    name: ModelName = ModelName.CONV1D_AUTOENCODER
    paradigm: Paradigm = Paradigm.RECONSTRUCTION
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


@dataclass(kw_only=True)
class TransformerEncoderConfig(ModelConfig):
    """Hyperparameters of the transformer autoencoder.

    Attention is bidirectional here, which is what reconstruction wants: the
    model may look at the whole window to rebuild it. That also means nothing
    stops it learning the identity, so latent_dim is what forces it to
    compress -- without the bottleneck every window reconstructs perfectly
    and the score is flat.
    """

    name: ModelName = ModelName.TRANSFORMER_ENCODER
    paradigm: Paradigm = Paradigm.RECONSTRUCTION
    patch_size: int = 16
    d_model: int = 128
    n_heads: int = 4
    num_layers: int = 3
    dim_feedforward: int = 256
    latent_dim: int = 32
    dropout: float = 0.0


@dataclass(kw_only=True)
class LSTMConfig(ModelConfig):
    """Hyperparameters of the LSTM forecaster."""

    name: ModelName = ModelName.LSTM
    paradigm: Paradigm = Paradigm.FORECASTING
    horizon: int = 1  # how far ahead each position predicts
    # the first positions have too little prefix to predict, so their errors
    # are large for every window and only add a constant to the score
    warmup: int = 0
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.0  # ignored by torch when num_layers == 1


@dataclass(kw_only=True)
class TransformerEncoderDecoderConfig(ModelConfig):
    """Hyperparameters of the causal transformer encoder-decoder forecaster.

    Attention over raw samples is quadratic, so a window is cut into patches
    first: 2048 samples at patch_size 16 is 128 tokens instead of 2048. Each
    token predicts the patch after it, which is why horizon must equal
    patch_size for the targets to line up.
    """

    name: ModelName = ModelName.TRANSFORMER_ENCODER_DECODER
    paradigm: Paradigm = Paradigm.FORECASTING
    horizon: int = 16  # must equal patch_size
    warmup: int = 0
    patch_size: int = 16
    d_model: int = 128
    n_heads: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.0


@dataclass
class TrainConfig:
    """Training loop configuration."""

    epochs: int
    learning_rate: float
    weight_decay: float
    patience: int  # early stopping; 0 disables it
    device: str
    seed: int
    checkpoint_name: str


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""

    channel: ImsChannel
    window_size: int
    hop: int  # stride between windows; hop == window_size means no overlap
    val_fraction: float  # fraction of healthy head to use for validation
    in_channels: int
    batch_size: int
    num_workers: int
    seed: int


@dataclass
class EvalConfig:
    """Every tunable knob of the evaluation lives here."""

    aggregation: Aggregation
    quantile: float  # threshold percentile on healthy validation scores
    output_artifacts: str

    def __post_init__(self) -> None:
        self.aggregation = Aggregation(self.aggregation)


@dataclass
class Config:
    """The whole config file: one field per top-level YAML section."""

    model_config: ModelConfig
    training_config: TrainConfig
    data_config: DataConfig
    evaluation_config: EvalConfig


class ConfigLoader:
    """Class containing static methods to load a config file into the dataclasses."""

    @staticmethod
    def build_model_config(raw_dict: dict[str, Any]) -> ModelConfig:
        """Pick the subclass whose default name matches the YAML."""
        values = dict(raw_dict)
        # the YAML carries the name as a string, the configs as an enum
        name = ModelName(values.pop('name'))
        for config_class in ModelConfig.__subclasses__():
            if config_class.__dataclass_fields__['name'].default == name:
                return config_class(name=name, **values)
        raise ValueError(f'unknown model {name.value!r}')

    @staticmethod
    def load_config(path: Path) -> Config:
        """Read the YAML file into the config dataclasses."""
        raw_dict = yaml.safe_load(path.read_text(encoding='utf-8'))
        return Config(
            model_config=ConfigLoader.build_model_config(
                raw_dict=raw_dict['model']
            ),
            training_config=TrainConfig(**raw_dict['training']),
            data_config=DataConfig(**raw_dict['data']),
            evaluation_config=EvalConfig(**raw_dict['evaluation']),
        )
