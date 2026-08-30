from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from torch import Tensor
    from torch.utils.data import DataLoader as TorchDataLoader
    from torch.utils.tensorboard import SummaryWriter

    from phm_101.data_types.models import TrainResult


class Detector(ABC):
    """
    Abstract base class for anomaly detectors.

    It provides base interfaces for a detector to be trained and evaluated on a batch of windows.
    Subclasses should implement the `fit` and `score_batch` methods.
    """

    @abstractmethod
    def fit(
        self,
        train_loader: TorchDataLoader,
        val_loader: TorchDataLoader,
        writer: SummaryWriter | None = None,
    ) -> TrainResult:
        """Fit the detector to healthy training data, optionally using a validation set for early stopping."""
        raise NotImplementedError()

    @abstractmethod
    def score_batch(self, windows: Tensor) -> np.ndarray:
        """Evaluate the detector on a batch of windows, returning one score per window."""
        raise NotImplementedError()

    @abstractmethod
    def state(self) -> dict[str, Tensor]:
        """Everything the detector needs to score again after a reload.

        Kept on the detector rather than reached for as `detector.model`, so a
        detector without a torch module can still be checkpointed.
        """
        raise NotImplementedError()

    @abstractmethod
    def load_state(self, state: dict[str, Tensor]) -> None:
        """Restore what state() returned."""
        raise NotImplementedError()
