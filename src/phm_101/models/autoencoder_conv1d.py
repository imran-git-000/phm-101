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
        c = self.config
        if c.window_size % 2**c.n_blocks:
            raise ValueError('window_size must be divisible by 2 ** n_blocks')

        padding = c.kernel_size // 2
        flat_dim = c.channels[-1] * c.bottleneck_length

        sizes = (c.in_channels,) + c.channels
        self.encoder = nn.Sequential(
            *[
                layer
                for i in range(c.n_blocks)
                for layer in self._block(
                    nn.Conv1d(
                        sizes[i],
                        sizes[i + 1],
                        c.kernel_size,
                        stride=2,
                        padding=padding,
                    ),
                    sizes[i + 1],
                )
            ]
        )
        self.to_latent = nn.Sequential(
            nn.Flatten(), nn.Linear(flat_dim, c.latent_dim)
        )
        self.from_latent = nn.Linear(c.latent_dim, flat_dim)

        reversed_sizes = sizes[::-1]
        self.decoder = nn.Sequential(
            *[
                layer
                for i in range(c.n_blocks)
                for layer in self._block(
                    nn.ConvTranspose1d(
                        reversed_sizes[i],
                        reversed_sizes[i + 1],
                        c.kernel_size,
                        stride=2,
                        padding=padding,
                        output_padding=1,
                    ),
                    reversed_sizes[i + 1],
                    # last block outputs the signal: no norm, no activation
                    final=i == c.n_blocks - 1,
                )
            ]
        )

    def _block(
        self, conv: nn.Module, out_channels: int, final: bool = False
    ) -> list[nn.Module]:
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
        c = self.config
        h = self.from_latent(z).view(-1, c.channels[-1], c.bottleneck_length)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-window MSE, the anomaly score. Returns (batch,)."""
        return ((self(x) - x) ** 2).mean(dim=(1, 2))
