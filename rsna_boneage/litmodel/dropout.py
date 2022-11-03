from typing import Any

import torch
from torch import vstack

from rsna_boneage.data import undo_boneage_rescale
from rsna_boneage.litmodel.base import LitRSNABoneage, LitRSNABoneageVarianceNet
from rsna_boneage.net.inception import RSNABoneageInceptionNetWithGender
from rsna_boneage.net.resnet import RSNABoneageResNetWithGender
from uncertainty.model import UncertaintyAwareModel


class LitRSNABoneageMCDropout(UncertaintyAwareModel, LitRSNABoneage):

    def __init__(self, *args, n_samples: int = 100, **kwargs):
        self.n_samples = n_samples
        super().__init__(*args, **kwargs)

    def forward_with_uncertainty(self, batch) -> tuple[torch.Tensor, Any]:
        # Enable Dropout Layers in Network for MC
        self.net.dropout.train()
        # Workarround for Gender Nets -> TODO: Abstraction into Interface
        if isinstance(self.net, RSNABoneageInceptionNetWithGender):
            self.net.inception.dropout.train()
        if isinstance(self.net, RSNABoneageResNetWithGender):
            self.net.resnet.dropout.train()

        with torch.no_grad():
            preds = [self.forward(batch).cpu() for _ in range(self.n_samples)]

        if self.undo_boneage_rescale:
            preds = [undo_boneage_rescale(pred) for pred in preds]
        preds = vstack(preds)

        # TODO: check return type of tensor
        # TODO: batch support
        # TODO: gaussian quantile thing calculation
        quantile_5 = preds.quantile(0.05)
        quantile_17 = preds.quantile(0.17)
        quantile_83 = preds.quantile(0.83)
        quantile_95 = preds.quantile(0.95)
        metrics = {
            'mean': preds.mean(),
            'median': preds.median(),
            'std': preds.std(),
            'var': preds.var(),
            'quantile_5': quantile_5,
            'quantile_17': quantile_17,
            'quantile_83': quantile_83,
            'quantile_95': quantile_95,
            'uncertainty': preds.std(),
            'predictions': preds,
        }
        return torch.Tensor([metrics['mean'], metrics['uncertainty']]), metrics


class LitRSNABoneageVarianceNetMCDropout(LitRSNABoneageVarianceNet):

    def __init__(self, n_samples: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.n_samples = n_samples

    def forward_with_uncertainty(self, input) -> tuple[torch.Tensor, Any]:
        # Enable Dropout Layers in Network for MC
        self.net.dropout.train()
        # Workarround for Gender Nets -> TODO: Abstraction into Interface
        if isinstance(self.net, RSNABoneageInceptionNetWithGender):
            self.net.inception.dropout.train()
        if isinstance(self.net, RSNABoneageResNetWithGender):
            self.net.resnet.dropout.train()

        with torch.no_grad():
            preds = [self.forward(input).cpu() for _ in range(self.n_samples)]

        if self.undo_boneage_rescale:
            preds = [undo_boneage_rescale(pred) for pred in preds]
        preds = vstack(preds)

        # TODO: check return type of tensor
        # TODO: batch support
        # TODO: gaussian quantile thing calculation
        return preds.mean(dim=0), None
