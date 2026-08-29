from typing import TYPE_CHECKING

import torch
from torch import nn

from phm_101.data_types.enums import ModelName
from phm_101.models.model_registry import ModelRegistry

if TYPE_CHECKING:
    from phm_101.config.configs import Conv1DAutoencoderConfig, DataConfig


@ModelRegistry.register(ModelName.CONV1D_AUTOENCODER)
class ConvAutoencoder1d(nn.Module):
    """1D convolutional autoencoder for unsupervised anomaly detection.

    Input and output are (batch, in_channels, window_size).
    """

    def __init__(
        self, model_config: Conv1DAutoencoderConfig, data_config: DataConfig
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.data_config = data_config

        if self.data_config.window_size % 2**self.model_config.n_blocks:
            raise ValueError('window_size must be divisible by 2 ** n_blocks')
        if self.model_config.kernel_size % 2 == 0:
            raise ValueError(
                'kernel_size must be odd, otherwise padding = kernel_size // 2'
                ' does not halve and double the length exactly'
            )

        # sequence length after all stride-2 blocks
        self.bottleneck_length = (
            self.data_config.window_size // 2**self.model_config.n_blocks
        )

        padding = self.model_config.kernel_size // 2
        # batch norm subtracts the mean, which cancels any preceding conv bias
        bias = not self.model_config.batch_norm
        flat_dim = self.model_config.channels[-1] * self.bottleneck_length

        sizes = (self.data_config.in_channels,) + self.model_config.channels
        self.encoder = nn.Sequential(
            *[
                layer
                for i in range(self.model_config.n_blocks)
                for layer in self._block(
                    nn.Conv1d(
                        sizes[i],
                        sizes[i + 1],
                        self.model_config.kernel_size,
                        stride=2,
                        padding=padding,
                        bias=bias,
                    ),
                    sizes[i + 1],
                )
            ]
        )
        self.to_latent = nn.Sequential(
            nn.Flatten(), nn.Linear(flat_dim, self.model_config.latent_dim)
        )
        # the ReLU stops this linear and the first decoder conv from
        # collapsing into a single linear map
        self.from_latent = nn.Sequential(
            nn.Linear(self.model_config.latent_dim, flat_dim),
            nn.ReLU(inplace=True),
        )

        reversed_sizes = sizes[::-1]
        self.decoder = nn.Sequential(
            *[
                layer
                for i in range(self.model_config.n_blocks)
                for layer in self._block(
                    nn.ConvTranspose1d(
                        reversed_sizes[i],
                        reversed_sizes[i + 1],
                        self.model_config.kernel_size,
                        stride=2,
                        padding=padding,
                        output_padding=1,
                        # no norm follows the last block, so it keeps its bias
                        bias=bias or i == self.model_config.n_blocks - 1,
                    ),
                    reversed_sizes[i + 1],
                    # last block outputs the signal: no norm, no activation
                    final=i == self.model_config.n_blocks - 1,
                )
            ]
        )

    def _block(
        self, conv: nn.Module, out_channels: int, final: bool = False
    ) -> list[nn.Module]:
        """Conv block made of conv, batch norm, ReLU, and dropout. The last block has no batch norm, ReLU, or dropout."""
        if final:
            return [conv]
        layers: list[nn.Module] = [conv]
        if self.model_config.batch_norm:
            layers.append(nn.BatchNorm1d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        if self.model_config.dropout:
            layers.append(nn.Dropout(self.model_config.dropout))
        return layers

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_latent(self.encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_latent(z).view(
            z.shape[0],
            self.model_config.channels[-1],
            self.bottleneck_length,
        )
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
