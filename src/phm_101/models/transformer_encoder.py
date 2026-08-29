from typing import TYPE_CHECKING

import torch
from torch import nn

from phm_101.data_types.enums import ModelName
from phm_101.models.model_registry import ModelRegistry

if TYPE_CHECKING:
    from torch import Tensor

    from phm_101.config.configs import DataConfig, TransformerEncoderConfig


@ModelRegistry.register(ModelName.TRANSFORMER_ENCODER)
class TransformerAutoencoder(nn.Module):
    """Transformer autoencoder: rebuilds the input signal.

    Input and output are both (batch, in_channels, window_size).
    """

    def __init__(
        self,
        model_config: TransformerEncoderConfig,
        data_config: DataConfig,
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.data_config = data_config

        patch_size = model_config.patch_size
        if data_config.window_size % patch_size:
            raise ValueError(
                f'window_size ({data_config.window_size}) must divide by '
                f'patch_size ({patch_size})'
            )
        self.patch_size = patch_size
        self.n_tokens = data_config.window_size // patch_size
        patch_dim = data_config.in_channels * patch_size
        flat_dim = self.n_tokens * model_config.d_model

        self.embedding = nn.Linear(patch_dim, model_config.d_model)
        # learned, one per token position
        self.positions = nn.Parameter(
            torch.zeros(1, self.n_tokens, model_config.d_model)
        )
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=model_config.d_model,
                nhead=model_config.n_heads,
                dim_feedforward=model_config.dim_feedforward,
                dropout=model_config.dropout,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=model_config.num_layers,
        )
        # the bottleneck: everything the model keeps about a window has to
        # pass through latent_dim numbers
        self.to_latent = nn.Sequential(
            nn.Flatten(), nn.Linear(flat_dim, model_config.latent_dim)
        )
        self.from_latent = nn.Sequential(
            nn.Linear(model_config.latent_dim, flat_dim),
            nn.GELU(),
        )
        self.head = nn.Linear(model_config.d_model, patch_dim)

    def encode(self, x: Tensor) -> Tensor:
        batch, channels, length = x.shape
        tokens = length // self.patch_size
        # (batch, channels, length) -> (batch, tokens, channels * patch_size)
        patched = (
            x.reshape(batch, channels, tokens, self.patch_size)
            .permute(0, 2, 1, 3)
            .reshape(batch, tokens, -1)
        )
        embedded = self.embedding(patched) + self.positions[:, :tokens]
        return self.to_latent(self.encoder(embedded))

    def decode(self, latent: Tensor, shape: torch.Size) -> Tensor:
        batch, channels, length = shape
        tokens = length // self.patch_size
        hidden = self.from_latent(latent).view(
            batch, tokens, self.model_config.d_model
        )
        return (
            self.head(hidden)
            .reshape(batch, tokens, channels, self.patch_size)
            .permute(0, 2, 1, 3)
            .reshape(batch, channels, length)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x), x.shape)
