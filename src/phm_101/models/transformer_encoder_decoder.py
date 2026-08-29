from typing import TYPE_CHECKING

import torch
from torch import nn

from phm_101.data_types.enums import ModelName
from phm_101.models.model_registry import ModelRegistry

if TYPE_CHECKING:
    from torch import Tensor

    from phm_101.config.configs import (
        DataConfig,
        TransformerEncoderDecoderConfig,
    )


@ModelRegistry.register(ModelName.TRANSFORMER_ENCODER_DECODER)
class TransformerEncoderDecoderForecaster(nn.Module):
    """Encoder-decoder transformer predicting each patch from the ones before.
    Input and output are both (batch, in_channels, length).
    """

    # register_buffer erases the type, so declare it for the checker
    causal_mask: Tensor

    def __init__(
        self,
        model_config: TransformerEncoderDecoderConfig,
        data_config: DataConfig,
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.data_config = data_config

        patch_size = model_config.patch_size
        if model_config.horizon != patch_size:
            raise ValueError(
                f'horizon must equal patch_size so each token predicts the '
                f'next patch, got horizon {model_config.horizon} and '
                f'patch_size {patch_size}'
            )
        # the detector feeds windows shortened by the horizon
        length = data_config.window_size - model_config.horizon
        if length % patch_size:
            raise ValueError(
                f'window_size - horizon ({length}) must divide by patch_size '
                f'({patch_size})'
            )
        self.patch_size = patch_size
        self.n_tokens = length // patch_size
        patch_dim = data_config.in_channels * patch_size

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
            num_layers=model_config.num_encoder_layers,
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=model_config.d_model,
                nhead=model_config.n_heads,
                dim_feedforward=model_config.dim_feedforward,
                dropout=model_config.dropout,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=model_config.num_decoder_layers,
        )
        self.head = nn.Linear(model_config.d_model, patch_dim)
        # registered so it moves with .to(device) and is not rebuilt per batch
        self.register_buffer(
            'causal_mask',
            nn.Transformer.generate_square_subsequent_mask(self.n_tokens),
            persistent=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, length = x.shape
        tokens = length // self.patch_size
        # (batch, channels, length) -> (batch, tokens, channels * patch_size)
        patched = (
            x.reshape(batch, channels, tokens, self.patch_size)
            .permute(0, 2, 1, 3)
            .reshape(batch, tokens, -1)
        )
        embedded = self.embedding(patched) + self.positions[:, :tokens]

        mask = self.causal_mask[:tokens, :tokens]
        memory = self.encoder(embedded, mask=mask, is_causal=True)
        decoded = self.decoder(
            embedded,
            memory,
            tgt_mask=mask,
            memory_mask=mask,
            tgt_is_causal=True,
            memory_is_causal=True,
        )
        # back to (batch, channels, length)
        return (
            self.head(decoded)
            .reshape(batch, tokens, channels, self.patch_size)
            .permute(0, 2, 1, 3)
            .reshape(batch, channels, length)
        )
