from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from sklearn.metrics import confusion_matrix, roc_auc_score

from phm_101.data_types.enums import Aggregation
from phm_101.data_types.models import EvalResult, Metrics

if TYPE_CHECKING:
    from torch.utils.data import DataLoader as TorchDataLoader

    from phm_101.config.configs import EvalConfig
    from phm_101.detectors.base import Detector

class Evaluator:
    """Pool a detector's window scores and measure detection quality."""

    def __init__(
        self,
        eval_config: EvalConfig,
        detector: Detector,
        val_dataloader: TorchDataLoader,
        test_dataloader: TorchDataLoader,
    ) -> None:
        self.logger = logger.bind(class_name=self.__class__.__name__)
        self.detector = detector
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.aggregation = eval_config.aggregation
        self.quantile = eval_config.quantile

    def evaluate(self) -> EvalResult:
        """Calibrate the threshold on healthy val data, then score the test set."""
        val_scores, _ = self.score(detector=self.detector, dataloader=self.val_dataloader)
        threshold = float(np.quantile(val_scores, self.quantile))

        scores, labels = self.score(detector=self.detector, dataloader=self.test_dataloader)
        predictions = (scores > threshold).astype('int8')
        metrics = self.metrics(labels, predictions, scores)

        self.logger.info('Evaluation: {metrics}', metrics=metrics)
        return EvalResult(
            scores=scores, labels=labels, threshold=threshold, metrics=metrics
        )

    def score(
        self, detector: Detector, dataloader: TorchDataLoader
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-snapshot anomaly scores and labels."""
        errors, snapshots, labels = [], [], []
        for windows, batch_labels, batch_snapshots in dataloader:
            errors.append(detector.score_batch(windows=windows))
            snapshots.append(batch_snapshots.numpy())
            labels.append(batch_labels.numpy())

        errors = np.concatenate(errors)
        snapshots = np.concatenate(snapshots)
        labels = np.concatenate(labels)
        return self._aggregate(errors, snapshots), self._aggregate_labels(
            labels, snapshots
        )

    @staticmethod
    def metrics(
        labels: np.ndarray, predictions: np.ndarray, scores: np.ndarray
    ) -> Metrics:
        """Counts at the chosen threshold, plus the threshold-free auroc."""
        tn, fp, fn, tp = confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).ravel()
        return Metrics(
            accuracy=(tp + tn) / (tp + tn + fp + fn),
            precision=tp / (tp + fp),
            recall=tp / (tp + fn),
            fpr=fp / (fp + tn),
            tpr=tp / (tp + fn),
            # undefined when the channel never fails, which is most of them
            auroc=(
                float(roc_auc_score(labels, scores))
                if labels.min() != labels.max()
                else float('nan')
            ),
        )

    def _aggregate(
        self, errors: np.ndarray, snapshots: np.ndarray
    ) -> np.ndarray:
        """Pool window errors into one score per snapshot."""
        n = int(snapshots.max()) + 1
        if self.aggregation is Aggregation.MEAN:
            counts = np.bincount(snapshots, minlength=n)
            return np.bincount(snapshots, weights=errors, minlength=n) / counts
        if self.aggregation is Aggregation.MAX:
            out = np.zeros(n, dtype=errors.dtype)
            np.maximum.at(out, snapshots, errors)
            return out
        raise ValueError(f'unknown aggregation {self.aggregation!r}')

    @staticmethod
    def _aggregate_labels(
        labels: np.ndarray, snapshots: np.ndarray
    ) -> np.ndarray:
        """All windows of a snapshot share its label, so take the first seen."""
        n = int(snapshots.max()) + 1
        out = np.zeros(n, dtype='int8')
        out[snapshots] = labels
        return out
