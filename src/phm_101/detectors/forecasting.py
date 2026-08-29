from typing import TYPE_CHECKING

import torch
from loguru import logger

from phm_101.detectors.base import Detector
from phm_101.ml_pipeline.train_pipeline import Trainer
from phm_101.models.model_registry import ModelRegistry
from phm_101.utils.utils import resolve_device

if TYPE_CHECKING:
    import numpy as np
    from torch import Tensor
    from torch.utils.data import DataLoader as TorchDataLoader

    from phm_101.config.configs import (
        DataConfig,
        ModelConfig,
        TrainConfig,
    )
    from phm_101.data_types.models import TrainResult


class ForecastingDetector(Detector):
    """
    Implementation of forecasting based methods for anomaly detectors.

    We use a dense prediction approach, where every position of a window is predicted from its own prefix.
    Every point of a window is predicted from its own prefix in one pass:
    x[0 .. T-h-1] -> x[h .. T-1].
    """

    def __init__(
        self,
        model_config: ModelConfig,
        data_config: DataConfig,
        train_config: TrainConfig,
    ) -> None:
        self.logger = logger.bind(class_name=self.__class__.__name__)

        self.model_config = model_config
        self.data_config = data_config
        self.train_config = train_config

        # how far ahead each position predicts
        self.horizon = model_config.horizon  # type: ignore

        self.device = resolve_device(train_config.device)
        self.model = ModelRegistry.get_model(
            model_config=self.model_config, data_config=self.data_config
        ).to(self.device)

    def split(self, windows: Tensor) -> tuple[Tensor, Tensor]:
        """Shift a batch of windows into (input, target) for dense prediction.

        Both come back as (batch, channels, window_size - horizon): position
        t of the input is asked for position t + horizon of the window, so
        every point is a target and the model reads only its own prefix.
        """
        return windows[..., : -self.horizon], windows[..., self.horizon :]

    def fit(
        self, train_loader: TorchDataLoader, val_loader: TorchDataLoader
    ) -> TrainResult:
        """Minimise prediction error on healthy windows."""
        trainer = Trainer(
            model=self.model,
            train_config=self.train_config,
            prepare=self.split,
        )
        result = trainer.train(
            train_dataloader=train_loader, val_dataloader=val_loader
        )
        # the trainer restored the weights of the best epoch
        self.model = trainer.model
        return result

    @torch.inference_mode()
    def score_batch(self, windows: Tensor) -> np.ndarray:
        """Per-window forecasting MSE, shaped (batch,)."""
        self.model.eval()
        inputs, targets = self.split(
            windows.to(self.device, non_blocking=True)
        )
        errors = (self.model(inputs) - targets) ** 2
        return errors.mean(dim=(1, 2)).cpu().numpy()
