from typing import Any, Optional

import torch
import torch.nn.functional
import torch.optim
from pytorch_lightning.core.module import LightningModule
from torch import Tensor, nn, squeeze
from torchvision.models.inception import InceptionOutputs
from rsna_boneage.data import undo_boneage_rescale

from uncertainty_fae.model import ADT_STAT_PREDS_VAR, TrainLoadMixin, UncertaintyAwareModel
from uncertainty_fae.util import nll_regression_loss
from uncertainty_fae.util import TrainConfig


class LitRSNABoneage(TrainLoadMixin, LightningModule):

    def __init__(
        self,
        net: nn.Module,
        lr: float = 3e-4,
        weight_decay: float = 0,
        momentum: float = 0,
        optim_type: str = 'adam',
        train_config: Optional[TrainConfig] = None,
        undo_boneage_rescale=False,
    ) -> None:
        super().__init__()

        self.net = net

        self.train_config = train_config
        self.optim_type = optim_type
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.mae = nn.L1Loss()
        self.mse = nn.MSELoss()

        self.undo_boneage_rescale = undo_boneage_rescale

        self.save_hyperparameters(ignore=['net'])

    def forward(self, x: Any) -> Tensor:
        logits = self.net.forward(x)

        # Workaround to make this LitModel usable with Inception, too.
        if isinstance(logits, InceptionOutputs):
            assert isinstance(logits.logits, Tensor)
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
    def load_model_from_disk(cls, checkpoint_path: str, **kwargs):
        return cls.load_from_checkpoint(checkpoint_path, **kwargs)

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

        # Create/get LR Scheduler for created Optimizer
        if isinstance(self.train_config, TrainConfig):
            lr_scheduler, monitor_metric = self.train_config.get_lr_scheduler(optim)

            if lr_scheduler:
                config['lr_scheduler'] = {'scheduler': lr_scheduler}
            if monitor_metric:
                config['lr_scheduler']['monitor'] = monitor_metric

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

    def forward_with_uncertainty(self, input) -> tuple[Tensor, Tensor, Optional[dict]]:
        pred_mean_var = self.forward(input)
        pred_mean = pred_mean_var[:, :1].cpu().flatten()
        pred_var = pred_mean_var[:, 1:].cpu().flatten()
        pred_std = torch.sqrt(pred_var)

        if self.undo_boneage_rescale:
            pred_mean = undo_boneage_rescale(pred_mean)
            pred_var = undo_boneage_rescale(pred_var)
            pred_std = undo_boneage_rescale(pred_std)

        metrics = {
            ADT_STAT_PREDS_VAR: pred_var,
        }
        return pred_mean, pred_std, metrics
