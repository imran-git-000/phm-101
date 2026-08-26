import torch
from torch import nn

from phm_101.data_types.models import AutoencoderConfig


class ConvAutoencoder1d(nn.Module):
    """1D convolutional autoencoder for unsupervised anomaly detection.

    Input and output are (batch, in_channels, window_size).
    """

    def __init__(self, config: AutoencoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or AutoencoderConfig()
        if self.config.window_size % 2**self.config.n_blocks:
            raise ValueError('window_size must be divisible by 2 ** n_blocks')
        if self.config.kernel_size % 2 == 0:
            raise ValueError(
                'kernel_size must be odd, otherwise padding = kernel_size // 2'
                ' does not halve and double the length exactly'
            )

        padding = self.config.kernel_size // 2
        # batch norm subtracts the mean, which cancels any preceding conv bias
        bias = not self.config.batch_norm
        flat_dim = self.config.channels[-1] * self.config.bottleneck_length

        sizes = (self.config.in_channels,) + self.config.channels
        self.encoder = nn.Sequential(
            *[
                layer
                for i in range(self.config.n_blocks)
                for layer in self._block(
                    nn.Conv1d(
                        sizes[i],
                        sizes[i + 1],
                        self.config.kernel_size,
                        stride=2,
                        padding=padding,
                        bias=bias,
                    ),
                    sizes[i + 1],
                )
            ]
        )
        self.to_latent = nn.Sequential(
            nn.Flatten(), nn.Linear(flat_dim, self.config.latent_dim)
        )
        # the ReLU stops this linear and the first decoder conv from
        # collapsing into a single linear map
        self.from_latent = nn.Sequential(
            nn.Linear(self.config.latent_dim, flat_dim),
            nn.ReLU(inplace=True),
        )

        reversed_sizes = sizes[::-1]
        self.decoder = nn.Sequential(
            *[
                layer
                for i in range(self.config.n_blocks)
                for layer in self._block(
                    nn.ConvTranspose1d(
                        reversed_sizes[i],
                        reversed_sizes[i + 1],
                        self.config.kernel_size,
                        stride=2,
                        padding=padding,
                        output_padding=1,
                        # no norm follows the last block, so it keeps its bias
                        bias=bias or i == self.config.n_blocks - 1,
                    ),
                    reversed_sizes[i + 1],
                    # last block outputs the signal: no norm, no activation
                    final=i == self.config.n_blocks - 1,
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
        if self.config.batch_norm:
            layers.append(nn.BatchNorm1d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        if self.config.dropout:
            layers.append(nn.Dropout(self.config.dropout))
        return layers

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_latent(self.encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_latent(z).view(
            z.shape[0],
            self.config.channels[-1],
            self.config.bottleneck_length,
        )
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
