import copy
import math
from typing import TYPE_CHECKING

import torch
from loguru import logger
from torch import Tensor, nn
from tqdm.auto import tqdm

from phm_101.data_types.models import TrainConfig, TrainResult
from phm_101.utils.utils import set_seed

if TYPE_CHECKING:
    from torch.utils.data import DataLoader as TorchDataLoader


class Trainer:
    """Train a reconstruction model on healthy windows."""

    def __init__(
        self, model: nn.Module, config: TrainConfig | None = None
    ) -> None:
        self.logger = logger.bind(class_name=self.__class__.__name__)
        self.config = config or TrainConfig()
        self.train_result: TrainResult = TrainResult(
            train_losses=[], val_losses=[]
        )
        if self.config.seed is not None:
            # only covers training-time randomness such as dropout
            set_seed(self.config.seed)
        self.device = self._resolve_device(self.config.device)
        self.model = model.to(self.device)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def train_step(self, dataloader: TorchDataLoader) -> float:
        """One epoch of training. Returns the mean reconstruction loss."""
        self.model.train()
        total, n_windows = 0.0, 0
        for windows, _, _ in dataloader:
            # compute prediction and loss
            batch: Tensor = windows.to(self.device, non_blocking=True)
            loss: Tensor = self.loss_fn(self.model(batch), batch)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # weight by batch size: a partial last batch must count for less
            total += loss.item() * batch.shape[0]
            n_windows += batch.shape[0]
        return total / n_windows

    @torch.inference_mode()
    def val_step(self, dataloader: TorchDataLoader) -> float:
        """One pass over held-out healthy windows. Returns the mean loss."""
        self.model.eval()
        total, n_windows = 0.0, 0
        for windows, _, _ in dataloader:
            batch: Tensor = windows.to(self.device, non_blocking=True)
            loss: Tensor = self.loss_fn(self.model(batch), batch)
            # weight by batch size: eval loaders keep their partial batch
            total += loss.item() * batch.shape[0]
            n_windows += batch.shape[0]
        return total / n_windows

    def train(
        self,
        train_dataloader: TorchDataLoader,
        val_dataloader: TorchDataLoader,
    ) -> TrainResult:
        """Fit the model, keeping the weights with the lowest validation loss."""
        best_loss, best_state, waited = float('inf'), None, 0
        self.logger.info(
            'Start training for {epochs} epochs using loss function: {loss}, and optimizer: {optimizer}',
            epochs=self.config.epochs,
            loss=self.loss_fn.__class__.__name__,
            optimizer=self.optimizer.__class__.__name__,
        )
        for epoch in tqdm(range(1, self.config.epochs + 1), desc='training'):
            train_loss = self.train_step(train_dataloader)
            val_loss = self.val_step(val_dataloader)
            self.train_result.train_losses.append(train_loss)
            self.train_result.val_losses.append(val_loss)
            self.logger.info(
                'Epoch {epoch}: train_loss={train:.6f} val_loss={val:.6f}',
                epoch=epoch,
                train=train_loss,
                val=val_loss,
            )
            if not (math.isfinite(train_loss) and math.isfinite(val_loss)):
                # a non-finite loss never beats best_loss
                raise RuntimeError(
                    f'Training diverged at epoch {epoch}: '
                    f'train_loss={train_loss}, val_loss={val_loss}'
                )

            if val_loss < best_loss:
                best_loss, waited = val_loss, 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                waited += 1
                if self.config.patience and waited >= self.config.patience:
                    self.logger.info(
                        'Early stopping at epoch {epoch}', epoch=epoch
                    )
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.logger.info('Best validation loss: {loss:.6f}', loss=best_loss)
        return self.train_result

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device != 'auto':
            return torch.device(device)
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
