from typing import Optional

import torch
import torch.nn.functional
import torch.optim
from pytorch_lightning.core.module import LightningModule
from torch import Tensor, nn, squeeze
from torch.nn.modules.loss import _Loss
from clavicle_ct.data import undo_clavicle_age_rescale

from clavicle_ct.loss import AgeLoss_2_Unscaled
from uncertainty_fae.model import ForwardMetrics, TrainLoadMixin, UncertaintyAwareModel
from uncertainty_fae.util import TrainConfig, nll_regression_loss


class LitClavicle(TrainLoadMixin, LightningModule):
    def __init__(
        self,
        net: nn.Module,
        lr: float = 3e-4,
        loss: Optional[_Loss] = None,
        weight_decay: float = 0,
        undo_boneage_rescale: bool = False,
        train_config: Optional[TrainConfig] = None,
        optim_type: str = "adam",
    ) -> None:
        super().__init__()

        self.net = net
        self.lr = lr
        self.weight_decay = weight_decay
        if not loss:
            self.loss = AgeLoss_2_Unscaled()
        else:
            self.loss = loss
        self.err_metric = nn.L1Loss()

        self.save_hyperparameters(ignore=["loss", "net"])

        self.undo_boneage_rescale = undo_boneage_rescale
        self.train_config = train_config
        self.optim_type = optim_type

        self.validation_results = []
        self.test_results = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)

        # Calculate loss
        train_loss = self.loss(torch.squeeze(logits, dim=1), y)
        self.log("loss", train_loss)

        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)

        y_true_np = torch.flatten(y).cpu().numpy()[0]
        y_pred_np = torch.flatten(logits)[0].cpu().numpy()
        self.validation_results.append([y_true_np, y_pred_np])

        # Calculate loss
        val_loss = self.loss(torch.squeeze(logits, dim=1), y)
        self.log("val_loss", val_loss)

        # Calculate error
        mae = self.err_metric(torch.squeeze(logits, dim=1), y)
        self.log("val_error", mae)

        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)

        y_true_np = torch.flatten(y)[0].cpu().numpy()
        y_pred_np = torch.flatten(logits)[0].cpu().numpy()
        self.test_results.append([y_true_np, y_pred_np])

        # Calculate error
        mae = self.err_metric(torch.squeeze(logits, dim=1), y)
        self.log("test_error", mae)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    @classmethod
    def load_model_from_disk(cls, checkpoint_path: str, **kwargs):
        return cls.load_from_checkpoint(checkpoint_path, **kwargs)


class LitClavicleVarianceNet(UncertaintyAwareModel, LitClavicle):
    def __init__(self, *args, lr: float = 0.00001, **kwargs) -> None:
        super().__init__(*args, lr=lr, **kwargs)

    def forward(self, x):
        logits = super().forward(x)
        # Apply softplus on variance units
        return torch.concat([logits[:, :1], torch.nn.functional.softplus(logits[:, 1:])], dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits = self.forward(x)
        logits = squeeze(logits, dim=1)

        # Calculate MAE between mean neurons ("first column") and targets (ignore variance neurons)
        mae = self.err_metric(logits[:, :1], y.unsqueeze(1))
        self.log("train_error", mae)

        loss = nll_regression_loss(logits, y)
        self.log("loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits = self.forward(x)
        logits = squeeze(logits, dim=1)

        # Calculate MAE between mean neurons ("first column") and targets (ignore variance neurons)
        mae = self.err_metric(logits[:, :1], y.unsqueeze(1))
        self.log("val_error", mae)

        loss = nll_regression_loss(logits, y)
        self.log("val_loss", loss)

        return loss

    def forward_with_uncertainty(self, input) -> tuple[Tensor, Tensor, ForwardMetrics]:
        pred_mean_var = self.forward(input)
        pred_mean = pred_mean_var[:, :1].cpu().flatten()
        pred_var = pred_mean_var[:, 1:].cpu().flatten()
        pred_std = torch.sqrt(pred_var)

        if self.undo_boneage_rescale:
            pred_mean = undo_clavicle_age_rescale(pred_mean)
            pred_std = undo_clavicle_age_rescale(pred_std, with_shift=False)

        return pred_mean, pred_std, ForwardMetrics()
