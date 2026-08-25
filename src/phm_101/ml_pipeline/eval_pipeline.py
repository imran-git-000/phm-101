from typing import TYPE_CHECKING

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn

from phm_101.data_types.models import EvalResult

if TYPE_CHECKING:
    from torch.utils.data import DataLoader as TorchDataLoader


class Evaluator:
    """Score snapshots with reconstruction error and measure detection quality."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | str = 'auto',
        aggregation: str = 'mean',  # 'mean' or 'max' over a snapshot's windows
        quantile: float = 0.99,  # threshold percentile on healthy val scores
    ) -> None:
        self.logger = logger.bind(class_name=self.__class__.__name__)
        self.aggregation = aggregation
        self.quantile = quantile
        self.device = (
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if device == 'auto'
            else torch.device(device)
        )
        self.model = model.to(self.device)

    @torch.inference_mode()
    def score(
        self, dataloader: TorchDataLoader
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-snapshot anomaly scores and labels."""
        self.model.eval()
        errors, snapshots, labels = [], [], []
        for windows, batch_labels, batch_snapshots in dataloader:
            batch = windows.unsqueeze(1).to(self.device)
            reconstruction = self.model(batch)
            errors.append(
                ((reconstruction - batch) ** 2).mean(dim=(1, 2)).cpu().numpy()
            )
            snapshots.append(batch_snapshots.numpy())
            labels.append(batch_labels.numpy())

        errors = np.concatenate(errors)
        snapshots = np.concatenate(snapshots)
        labels = np.concatenate(labels)
        return self._aggregate(errors, snapshots), self._aggregate_labels(
            labels, snapshots
        )

    def evaluate(
        self,
        test_dataloader: TorchDataLoader,
        val_dataloader: TorchDataLoader,
    ) -> EvalResult:
        """Calibrate the threshold on healthy val data, then score the test set."""
        val_scores, _ = self.score(val_dataloader)
        threshold = float(np.quantile(val_scores, self.quantile))

        scores, labels = self.score(test_dataloader)
        predictions = (scores > threshold).astype('int8')

        metrics = {
            'threshold': threshold,
            'fault_rate': float(labels.mean()),
            'precision': float(
                precision_score(labels, predictions, zero_division=0)
            ),
            'recall': float(
                recall_score(labels, predictions, zero_division=0)
            ),
            'f1': float(f1_score(labels, predictions, zero_division=0)),
        }
        # undefined when the channel never fails
        if labels.min() != labels.max():
            metrics['auroc'] = float(roc_auc_score(labels, scores))
            metrics['average_precision'] = float(
                average_precision_score(labels, scores)
            )
            metrics['detection_delay'] = self._detection_delay(
                scores, labels, threshold
            )

        self.logger.info('Evaluation: {metrics}', metrics=metrics)
        return EvalResult(
            scores=scores, labels=labels, threshold=threshold, metrics=metrics
        )

    def _aggregate(
        self, errors: np.ndarray, snapshots: np.ndarray
    ) -> np.ndarray:
        """Pool window errors into one score per snapshot."""
        n = int(snapshots.max()) + 1
        if self.aggregation == 'mean':
            counts = np.bincount(snapshots, minlength=n)
            return np.bincount(snapshots, weights=errors, minlength=n) / counts
        if self.aggregation == 'max':
            out = np.zeros(n, dtype=errors.dtype)
            np.maximum.at(out, snapshots, errors)
            return out
        raise ValueError(
            f"aggregation must be 'mean' or 'max', got {self.aggregation!r}"
        )

    @staticmethod
    def _aggregate_labels(
        labels: np.ndarray, snapshots: np.ndarray
    ) -> np.ndarray:
        """All windows of a snapshot share its label, so take the first seen."""
        n = int(snapshots.max()) + 1
        out = np.zeros(n, dtype='int8')
        out[snapshots] = labels
        return out

    @staticmethod
    def _detection_delay(
        scores: np.ndarray, labels: np.ndarray, threshold: float
    ) -> float:
        """Snapshots between the labelled onset and the first alarm at or after it."""
        onset = int(np.argmax(labels == 1))
        alarms = np.flatnonzero(scores[onset:] > threshold)
        return float(alarms[0]) if alarms.size else float('inf')
