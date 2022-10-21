from typing import Any

import torch
import torch.nn.functional
import torch.optim
from pytorch_lightning.core.lightning import LightningModule
from torch import nn, squeeze
from torchvision.models.inception import InceptionOutputs

from rsna_boneage.data import RSNA_BONEAGE_DATASET_MAX_AGE, RSNA_BONEAGE_DATASET_MIN_AGE
from uncertainty.model import TrainMixin


class LitRSNABoneage(TrainMixin, LightningModule):

    def __init__(self, net: nn.Module, lr: float = 3e-4, weight_decay: float = 0,
                 undo_boneage_rescaling=False, **kwargs) -> None:
        super().__init__()

        self.net = net
        self.lr = lr

        self.mae = nn.L1Loss()
        self.mse = nn.MSELoss()

        self.weight_decay = weight_decay
        self.undo_boneage_rescaling = undo_boneage_rescaling

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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def _undo_rescale_boneage(self, boneage_rescaled):
        lower_bound = RSNA_BONEAGE_DATASET_MIN_AGE
        upper_bound = RSNA_BONEAGE_DATASET_MAX_AGE
        return (boneage_rescaled * (upper_bound - lower_bound)) + lower_bound
