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

    from phm_101.config.configs import DataConfig, ModelConfig, TrainConfig
    from phm_101.data_types.models import TrainResult


class ReconstructionDetector(Detector):
    """
    Implementation of reconstruction based methods for anomaly detectors.
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

        self.device = resolve_device(train_config.device)
        self.model = ModelRegistry.get_model(
            model_config=self.model_config, data_config=self.data_config
        ).to(self.device)

    def fit(
        self, train_loader: TorchDataLoader, val_loader: TorchDataLoader
    ) -> TrainResult:
        """Minimise reconstruction error on healthy windows."""
        trainer = Trainer(model=self.model, train_config=self.train_config)
        result = trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            is_forecasting=False,
        )
        # the trainer restored the weights of the best epoch
        self.model = trainer.model
        return result

    @torch.inference_mode()
    def score_batch(self, windows: Tensor) -> np.ndarray:
        """Per-window reconstruction MSE, shaped (batch,)."""
        self.model.eval()
        batch = windows.to(self.device, non_blocking=True)
        reconstruction = self.model(batch)
        # mean over channels and time leaves one score per window
        return ((reconstruction - batch) ** 2).mean(dim=(1, 2)).cpu().numpy()
