from typing import Any

import torch
import torch.nn.functional
import torch.optim
from pytorch_lightning.core.module import LightningModule
from torch import nn, squeeze
from torchvision.models.inception import InceptionOutputs

from uncertainty.model import TrainLoadMixin


class LitRSNABoneage(TrainLoadMixin, LightningModule):

    def __init__(self, net: nn.Module, lr: float = 3e-4, weight_decay: float = 0,
                 momentum: float = 0, optim_type: str = 'adam', undo_boneage_rescale=False):
        super().__init__()

        self.net = net

        self.optim_type = optim_type
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.mae = nn.L1Loss()
        self.mse = nn.MSELoss()

        self.undo_boneage_rescale = undo_boneage_rescale

        self.save_hyperparameters(ignore=['net'])

    def forward(self, x: Any) -> Any:
        logits = self.net.forward(x)

        # Workaround to make this LitModel usable with Inception, too.
        if isinstance(logits, InceptionOutputs):
            logits = logits.logits

        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits = self.forward(x)
        logits = squeeze(logits, dim=1)

        mae = self.mae(logits, y)
        self.log('mae', mae)

        loss = self.mse(logits, y)
        self.log('loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits = self.forward(x)
        logits = squeeze(logits, dim=1)

        mae = self.mae(logits, y)
        self.log('val_mae', mae)

        loss = self.mse(logits, y)
        self.log('val_loss', loss)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        # Forward pass
        logits = self.forward(x)
        logits = squeeze(logits, dim=1)

        mae = self.mae(logits, y)
        self.log('test_mae', mae)

        loss = self.mse(logits, y)
        self.log('test_loss', loss)

    @classmethod
    def load_model_from_disk(cls, **kwargs):
        return cls.load_from_checkpoint(**kwargs)

    def configure_optimizers(self):
        if self.optim_type == 'adam':
            return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optim_type == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                                   momentum=self.momentum)
        else:
            raise ValueError(f'Unkown optimizer type: {self.optim_type}')
