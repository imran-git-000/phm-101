import random
from pathlib import Path

import torch
from loguru import logger


def save_model(
    model: torch.nn.Module, target_dir: str, model_name: str
) -> None:
    """Saves a PyTorch model to a target directory.

    Args:
      model: A target PyTorch model to save.
      target_dir: A directory for saving the model to.
      model_name: A filename for the saved model. Should include
        either ".pth" or ".pt" as the file extension.

    Example usage:
      save_model(model=model_0,
                 target_dir="models",
                 model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    assert model_name.endswith(('.pth', '.pt'))
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    logger.info('Saving model to: {path}', path=model_save_path)
    torch.save(obj=model.state_dict(), f=model_save_path)


def set_seed(seed: int) -> None:
    """Seed python and torch so a run can be reproduced.

    Call this *before* building the model: a module initialises its weights
    in its constructor, so seeding afterwards leaves them unreproducible.
    torch.manual_seed covers the CUDA devices as well. Nothing here draws
    from numpy's global RNG; use np.random.default_rng(seed) if that changes.
    """
    random.seed(seed)
    torch.manual_seed(seed)
