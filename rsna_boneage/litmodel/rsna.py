
from typing import Any, Tuple

import torch
from torch import vstack
from rsna_boneage.litmodel import LitRSNABoneage
from rsna_boneage.net.inception import RSNABoneageInceptionNetWithGender
from rsna_boneage.net.resnet import RSNABoneageResNetWithGender
from uncertainty.model import UncertaintyAwareModel


class LitRSNABoneageMCDropout(UncertaintyAwareModel, LitRSNABoneage):

    def __init__(self, mc_iterations: int = 100, *args, **kwargs):
        self.mc_iterations = mc_iterations
        super().__init__(*args, **kwargs)

    def forward_with_uncertainty(self, batch) -> Tuple[torch.Tensor, Any]:
        # Enable Dropout Layers in Network for MC
        self.net.dropout.train()
        # Workarround for Gender Nets -> TODO: Abstraction into Interface
        if isinstance(self.net, RSNABoneageInceptionNetWithGender):
            self.net.inception.dropout.train()
        if isinstance(self.net, RSNABoneageResNetWithGender):
            self.net.resnet.dropout.train()

        with torch.no_grad():
            preds = [self.forward(batch).cpu() for _ in range(self.mc_iterations)]

        # TODO: rescaling param is not set
        if self.undo_boneage_rescaling:
            preds = [self._undo_rescale_boneage(pred) for pred in preds]
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
