import argparse
import random
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from loguru import logger

from phm_101.config.configs import ConfigLoader, DataConfig
from phm_101.data_types.enums import ImsChannel
from phm_101.data_types.models import Checkpoint

if TYPE_CHECKING:
    from phm_101.config.configs import ModelConfig


def _plain(config: ModelConfig | DataConfig) -> dict[str, object]:
    """A config as plain data.

    asdict leaves ModelName, Paradigm and ImsChannel as enum objects, which
    torch.load refuses to unpickle under weights_only=True. Their values are
    strings, and every config coerces them back in __post_init__.
    """
    return {
        key: value.value if isinstance(value, Enum) else value
        for key, value in asdict(config).items()
    }


def save_checkpoint(
    state: dict[str, torch.Tensor],
    model_config: ModelConfig,
    data_config: DataConfig,
    *,
    channel: ImsChannel,
    path: Path,
) -> None:
    """Write the weights and everything needed to score with them again.

    Only plain types go to disk -- tensors, dicts, floats, strings -- so the
    file loads back under weights_only=True and never unpickles arbitrary
    objects.
    """
    if path.suffix not in {'.pt', '.pth'}:
        raise ValueError(f'checkpoint must end in .pt or .pth, got {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info('Saving checkpoint to: {path}', path=path)
    torch.save(
        obj={
            'state_dict': state,
            'model_config': _plain(model_config),
            'data_config': _plain(data_config),
            'channel': ImsChannel(channel).value,
        },
        f=path,
    )


def load_checkpoint(path: Path) -> Checkpoint:
    """Read back what save_checkpoint wrote.

    The stored `name` decides which config class is rebuilt, so a checkpoint
    describes its own model family.
    """
    logger.info('Loading checkpoint from: {path}', path=path)
    raw = torch.load(path, map_location='cpu', weights_only=True)
    return Checkpoint(
        state_dict=raw['state_dict'],
        model_config=ConfigLoader.build_model_config(raw['model_config']),
        data_config=DataConfig(**raw['data_config']),
        channel=ImsChannel(raw['channel']),
    )


def resolve_device(device: str) -> torch.device:
    """Turn the configured device string into a torch device."""
    if device != 'auto':
        return torch.device(device)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed: int) -> None:
    """Seed python and torch so a run can be reproduced.

    Call this *before* building the model: a module initialises its weights
    in its constructor, so seeding afterwards leaves them unreproducible.
    torch.manual_seed covers the CUDA devices as well. Nothing here draws
    from numpy's global RNG; use np.random.default_rng(seed) if that changes.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='phm_101',
        description='Train and evaluate an unsupervised bearing fault detector.',
    )
    parser.add_argument(
        '--channel',
        default=None,
        help="channel to run, or 'all' for every IMS channel "
        '(default: the channel in the config file)',
    )
    parser.add_argument(
        '--model-path',
        type=Path,
        default=None,
        help='score with this checkpoint instead of training a new model',
    )
    return parser
