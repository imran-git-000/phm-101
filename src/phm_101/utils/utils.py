import argparse
import random
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from loguru import logger

from phm_101.config.configs import ConfigLoader
from phm_101.data_types.models import Checkpoint

if TYPE_CHECKING:
    from phm_101.config.configs import ModelConfig


def save_checkpoint(
    state: dict[str, torch.Tensor],
    config: ModelConfig,
    *,
    mean: float,
    std: float,
    channel: str,
    path: Path,
) -> None:
    """Write the weights and everything needed to score with them again.

    Only plain types go to disk -- tensors, a dict, floats, a string -- so the
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
            'config': asdict(config),
            'mean': mean,
            'std': std,
            'channel': channel,
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
        config=ConfigLoader.build_model_config(raw['config']),
        mean=float(raw['mean']),
        std=float(raw['std']),
        channel=str(raw['channel']),
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
