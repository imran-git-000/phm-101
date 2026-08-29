from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from torch import Tensor
    from torch.utils.data import DataLoader as TorchDataLoader

    from phm_101.data_types.models import TrainResult


class Detector(ABC):
    """
    Abstract base class for anomaly detectors.

    It provides base interfaces for a detector to be trained and evaluated on a batch of windows.
    Subclasses should implement the `fit` and `score_batch` methods.
    """

    @abstractmethod
    def fit(
        self, train_loader: TorchDataLoader, val_loader: TorchDataLoader
    ) -> TrainResult:
        """Fit the detector to healthy training data, optionally using a validation set for early stopping."""
        raise NotImplementedError()

    @abstractmethod
    def score_batch(self, windows: Tensor) -> np.ndarray:
        """Evaluate the detector on a batch of windows, returning one score per window."""
        raise NotImplementedError()
