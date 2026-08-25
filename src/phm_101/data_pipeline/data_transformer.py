from dataclasses import replace

import numpy as np
from loguru import logger

from phm_101.data_types.enums import ImsTests
from phm_101.data_types.models import ChannelData, ChannelSplit
from phm_101.utils.ims import test_of


class DataTransformer:
    """Split a channel chronologically and z-score with train statistics."""

    def __init__(self) -> None:
        self.logger = logger.bind(class_name=self.__class__.__name__)
        self.val_fraction = 0.2

        # last timestamp considered safely healthy
        self.train_end: dict[ImsTests, str] = {
            ImsTests.T1: '2003-11-18 19:46:07',
            ImsTests.T2: '2004-02-16 07:32:39',
            ImsTests.T3: '2004-04-15 23:42:55',
        }

        self.mean: float | None = None
        self.std: float | None = None

    def train_test_split(self, data: ChannelData) -> ChannelSplit:
        """Healthy head is train + val, everything after is test."""
        cutoff = np.datetime64(self.train_end[test_of(data.channel)])
        n_healthy = int((data.timestamps < cutoff).sum())
        n_train = n_healthy - int(n_healthy * self.val_fraction)

        self.logger.info(
            'Split {channel}: {train} train, {val} val, {test} test',
            channel=data.channel,
            train=n_train,
            val=n_healthy - n_train,
            test=len(data.timestamps) - n_healthy,
        )
        return ChannelSplit(
            train=self._slice(data=data, s=slice(None, n_train)),
            val=self._slice(data=data, s=slice(n_train, n_healthy)),
            test=self._slice(data=data, s=slice(n_healthy, None)),
        )

    def fit(self, data: ChannelData) -> None:
        """Compute z-score statistics on healthy signals."""
        self.mean = float(data.signals.mean())
        self.std = float(data.signals.std())
        self.logger.info(
            'Fitted mean={mean:.6f} std={std:.6f}',
            mean=self.mean,
            std=self.std,
        )

    def transform(self, data: ChannelData) -> ChannelData:
        """Apply the fitted statistics."""
        self._check_fitted()
        return replace(data, signals=(data.signals - self.mean) / self.std)

    def inverse_transform(self, data: ChannelData) -> ChannelData:
        """Undo the fitted statistics."""
        self._check_fitted()
        return replace(data, signals=data.signals * self.std + self.mean)

    def _check_fitted(self) -> None:
        if self.mean is None or self.std is None:
            raise ValueError('call fit() first')

    @staticmethod
    def _slice(data: ChannelData, s: slice) -> ChannelData:
        return replace(
            data,
            timestamps=data.timestamps[s],
            signals=data.signals[s],
            labels=data.labels[s],
        )
