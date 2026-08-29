from typing import TYPE_CHECKING

from torch import nn

from phm_101.data_types.enums import ModelName
from phm_101.models.model_registry import ModelRegistry

if TYPE_CHECKING:
    from torch import Tensor

    from phm_101.config.configs import DataConfig, LSTMConfig


@ModelRegistry.register(ModelName.LSTM)
class LSTMForecaster(nn.Module):
    """Predicts every point of a window from the points before it.

    Causal by construction: an LSTM reads left to right, so the output at
    step t depends only on steps <= t. Never make this bidirectional -- the
    model would read the future, drive the loss to nothing and leave every
    window scoring the same.

    Input and output are both (batch, in_channels, length).
    """

    def __init__(
        self, model_config: LSTMConfig, data_config: DataConfig
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.data_config = data_config

        self.lstm = nn.LSTM(
            input_size=data_config.in_channels,
            hidden_size=model_config.hidden_size,
            num_layers=model_config.num_layers,
            # torch only applies dropout between layers, and warns if asked
            # for it with a single one
            dropout=(
                model_config.dropout if model_config.num_layers > 1 else 0.0
            ),
            batch_first=True,
            bidirectional=False,
        )
        self.head = nn.Linear(
            model_config.hidden_size, data_config.in_channels
        )

    def forward(self, x: Tensor) -> Tensor:
        # (batch, channels, length) -> (batch, length, channels), which is
        # what batch_first expects
        hidden, _ = self.lstm(x.transpose(1, 2))
        return self.head(hidden).transpose(1, 2)
