from typing import Any, Optional, Tuple

import torch
import torch.nn.functional
import torch.optim
from pytorch_lightning.core.module import LightningModule
from torch import nn, squeeze
from torchvision.models.inception import InceptionOutputs

from uncertainty.model import TrainLoadMixin, UncertaintyAwareModel
from util.nll_regression_loss import nll_regression_loss


class LitRSNABoneage(TrainLoadMixin, LightningModule):

    def __init__(
        self,
        net: nn.Module,
        lr: float = 3e-4,
        weight_decay: float = 0,
        momentum: float = 0,
        optim_type: str = 'adam',
        lr_scheduler: Optional[str] = 'reduce_lr_on_plateau',
        undo_boneage_rescale=False
    ) -> None:
        super().__init__()

        self.net = net

        self.optim_type = optim_type
        self.lr_scheduler = lr_scheduler
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
            optim = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optim_type == 'sgd':
            optim = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                                    momentum=self.momentum)
        else:
            raise ValueError(f'Unkown optimizer type: {self.optim_type}')

        config = {
            'optimizer': optim
        }

        if self.lr_scheduler == 'reduce_lr_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim, factor=0.9, patience=10, threshold=1e-4)
            config['lr_scheduler'] = {'scheduler': scheduler, 'monitor': 'val_loss'}
        elif self.lr_scheduler:
            # an unknown scheduler is given
            raise ValueError(f'Unkown lr scheduler type: {self.lr_scheduler}')

        return config


class LitRSNABoneageVarianceNet(UncertaintyAwareModel, LitRSNABoneage):

    def __init__(self, *args, lr: float = 0.00001, **kwargs):
        super().__init__(*args, lr=lr, **kwargs)

    def forward(self, x):
        logits = super().forward(x)
        # Apply softplus on variance units
        return torch.concat([
            logits[:, :1],
            torch.nn.functional.softplus(logits[:, 1:])], dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits = self.forward(x)
        logits = squeeze(logits, dim=1)

        # Calculate MAE between mean neurons ("first column") and targets (ignore variance neurons)
        mae = self.mae(logits[:, :1], y.unsqueeze(1))
        self.log('val_mae', mae)

        loss = nll_regression_loss(logits, y)
        self.log('loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits = self.forward(x)
        logits = squeeze(logits, dim=1)

        # Calculate MAE between mean neurons ("first column") and targets (ignore variance neurons)
        mae = self.mae(logits[:, :1], y.unsqueeze(1))
        self.log('val_mae', mae)

        loss = nll_regression_loss(logits, y)
        self.log('val_loss', loss)

        return loss

    def forward_with_uncertainty(self, input) -> Tuple[torch.Tensor, Any]:
        return self.forward(input), None
