import copy
from typing import TYPE_CHECKING

import torch
from loguru import logger
from torch import nn
from tqdm.auto import tqdm

from phm_101.data_types.models import TrainConfig

if TYPE_CHECKING:
    from torch.utils.data import DataLoader as TorchDataLoader


class Trainer:
    """Train a reconstruction model on healthy windows."""

    def __init__(
        self, model: nn.Module, config: TrainConfig | None = None
    ) -> None:
        self.logger = logger.bind(class_name=self.__class__.__name__)
        self.config = config or TrainConfig()
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
        total = 0.0
        for windows, _, _ in dataloader:
            # (batch, window_size) -> (batch, 1, window_size) for Conv1d
            batch = windows.unsqueeze(1).to(self.device)
            loss = self.loss_fn(self.model(batch), batch)
            self.optimizer.zero_grad()
            loss.backward()
            if self.config.grad_clip:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )
            self.optimizer.step()
            total += loss.item()
        return total / len(dataloader)

    @torch.inference_mode()
    def val_step(self, dataloader: TorchDataLoader) -> float:
        """One pass over held-out healthy windows. Returns the mean loss."""
        self.model.eval()
        total = 0.0
        for windows, _, _ in dataloader:
            batch = windows.unsqueeze(1).to(self.device)
            total += self.loss_fn(self.model(batch), batch).item()
        return total / len(dataloader)

    def train(
        self,
        train_dataloader: TorchDataLoader,
        val_dataloader: TorchDataLoader,
    ) -> dict[str, list[float]]:
        """Fit the model, keeping the weights with the lowest validation loss."""
        history: dict[str, list[float]] = {'train_loss': [], 'val_loss': []}
        best_loss, best_state, waited = float('inf'), None, 0

        for epoch in tqdm(range(1, self.config.epochs + 1), desc='training'):
            train_loss = self.train_step(train_dataloader)
            val_loss = self.val_step(val_dataloader)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            self.logger.info(
                'Epoch {epoch}: train_loss={train:.6f} val_loss={val:.6f}',
                epoch=epoch,
                train=train_loss,
                val=val_loss,
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
        return history

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device != 'auto':
            return torch.device(device)
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
