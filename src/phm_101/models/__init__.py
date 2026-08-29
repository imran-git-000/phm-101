"""Model architectures.

Importing them here is what fills the ModelRegistry: the decorator only runs
when the module is imported, so a registry built from an empty package would
report every model as unknown.
"""

from phm_101.models.autoencoder_conv1d import ConvAutoencoder1d
from phm_101.models.lstm import LSTMForecaster
from phm_101.models.model_registry import ModelRegistry
from phm_101.models.transformer_encoder import TransformerAutoencoder
from phm_101.models.transformer_encoder_decoder import (
    TransformerEncoderDecoderForecaster,
)

__all__ = [
    'ConvAutoencoder1d',
    'LSTMForecaster',
    'ModelRegistry',
    'TransformerAutoencoder',
    'TransformerEncoderDecoderForecaster',
]
