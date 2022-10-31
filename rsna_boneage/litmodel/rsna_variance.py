from typing import Any, Tuple

import torch
from torch import squeeze
from torch import vstack
from rsna_boneage.data import undo_boneage_rescale

from rsna_boneage.litmodel import LitRSNABoneage
from rsna_boneage.net.inception import RSNABoneageInceptionNetWithGender
from rsna_boneage.net.resnet import RSNABoneageResNetWithGender
from uncertainty.model import UncertaintyAwareModel
from util.nll_regression_loss import nll_regression_loss


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

        # TODO: we can do this on the first neuron (mean neuron)
        # mae = self.mae(logits, y)
        # self.log('mae', mae)

        loss = nll_regression_loss(logits, y)
        self.log('loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits = self.forward(x)
        logits = squeeze(logits, dim=1)

        mae = self.mae(logits[:, :1], y.view(logits.shape[0], 1))
        self.log('val_mae', mae)

        loss = nll_regression_loss(logits, y)
        self.log('val_loss', loss)

        return loss

    def forward_with_uncertainty(self, input) -> Tuple[torch.Tensor, Any]:
        return self.forward(input), None


class LitRSNABoneageVarianceNetMCDropout(LitRSNABoneageVarianceNet):

    def __init__(self, mc_iterations: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.mc_iterations = mc_iterations

    def forward_with_uncertainty(self, input) -> Tuple[torch.Tensor, Any]:
        # Enable Dropout Layers in Network for MC
        self.net.dropout.train()
        # Workarround for Gender Nets -> TODO: Abstraction into Interface
        if isinstance(self.net, RSNABoneageInceptionNetWithGender):
            self.net.inception.dropout.train()
        if isinstance(self.net, RSNABoneageResNetWithGender):
            self.net.resnet.dropout.train()

        with torch.no_grad():
            preds = [self.forward(input).cpu() for _ in range(self.mc_iterations)]

        if self.undo_boneage_rescale:
            preds = [undo_boneage_rescale(pred) for pred in preds]
        preds = vstack(preds)

        # TODO: check return type of tensor
        # TODO: batch support
        # TODO: gaussian quantile thing calculation
        return preds.mean(dim=0), None
