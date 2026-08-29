from enum import Enum


class ImsTest(Enum):
    """IMS bearing tests.

    One parquet file per accelerometer, named ims_bearing_t1b3_x.
    """

    T1 = '1st_test'
    T2 = '2nd_test'
    T3 = '3rd_test'


class ImsChannel(Enum):
    """Accelerometer channels in the IMS bearing tests."""

    T1B1x = 't1b1x'
    T1B1y = 't1b1y'
    T1B2x = 't1b2x'
    T1B2y = 't1b2y'
    T1B3x = 't1b3x'
    T1B3y = 't1b3y'
    T1B4x = 't1b4x'
    T1B4y = 't1b4y'
    T2B1 = 't2b1'
    T2B2 = 't2b2'
    T2B3 = 't2b3'
    T2B4 = 't2b4'
    T3B1 = 't3b1'
    T3B2 = 't3b2'
    T3B3 = 't3b3'
    T3B4 = 't3b4'


class Aggregation(Enum):
    """How to aggregate window-level scores into a snapshot-level score."""

    MEAN = 'mean'
    MAX = 'max'


class ModelName(Enum):
    """The name of the model to use."""

    PCA = 'pca'
    CONV1D_AUTOENCODER = 'conv1d_autoencoder'
    LSTM = 'lstm'
    TRANSFORMER_ENCODER = 'transformer_encoder'
    TRANSFORMER_ENCODER_DECODER = 'transformer_encoder_decoder'


class Paradigm(Enum):
    """The paradigm of the model to use."""

    FORECASTING = 'forecasting'
    RECONSTRUCTION = 'reconstruction'
