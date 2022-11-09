from typing import Any, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from rsna_boneage.data import undo_boneage_rescale
from rsna_boneage.litmodel.base import LitRSNABoneage, LitRSNABoneageVarianceNet
from uncertainty.model import (ADT_STAT_PREDS_DISTINCT, ADT_STAT_PREDS_VAR, UncertaintyAwareModel,
                               uam_evaluate_dataset_default)
from util import dropout_train


class LitRSNABoneageMCDropout(UncertaintyAwareModel, LitRSNABoneage):

    def __init__(self, *args, n_samples: int = 100, **kwargs):
        self.n_samples = n_samples
        super().__init__(*args, **kwargs)

    def forward_with_uncertainty(self, batch) -> tuple[Tensor, Tensor, Optional[dict[str, Any]]]:
        # Enable Dropout Layers in Network for MC
        self.apply(dropout_train)

        with torch.no_grad():
            preds = [self.forward(batch).cpu().flatten() for _ in range(self.n_samples)]
            preds = torch.stack(preds)

        preds_mean = preds.mean(dim=0)
        preds_var = preds.var(dim=0)
        preds_std = preds.std(dim=0)

        if self.undo_boneage_rescale:
            preds = undo_boneage_rescale(preds)
            preds_mean = undo_boneage_rescale(preds_mean)
            preds_var = undo_boneage_rescale(preds_var)
            preds_std = undo_boneage_rescale(preds_std)

        metrics = {
            ADT_STAT_PREDS_DISTINCT: [preds[:, i:i+1].flatten() for i in range(len(batch))],
            ADT_STAT_PREDS_VAR: preds_var,
        }
        return preds_mean, preds_std, metrics

    def evaluate_dataset(
        self, dataloader: DataLoader
    ) -> tuple[Any, Tensor, Tensor, Tensor, Tensor, Optional[dict[str, Any]]]:
        self.eval()
        self.cuda()
        return uam_evaluate_dataset_default(self, self.device, dataloader)


class LitRSNABoneageVarianceNetMCDropout(LitRSNABoneageVarianceNet):

    def __init__(self, *args, n_samples: int = 100, **kwargs):
        self.n_samples = n_samples
        super().__init__(*args, **kwargs)

    def forward_with_uncertainty(self, batch) -> tuple[Tensor, Tensor, Optional[dict[str, Any]]]:
        # Enable Dropout Layers in Network for MC
        self.apply(dropout_train)

        with torch.no_grad():
            iter_means = []
            iter_vars = []

            for _ in range(self.n_samples):
                pred_mean_var = self.forward(batch).cpu()
                iter_means.append(pred_mean_var[:, :1].cpu().flatten())  # Mean Column (1st)
                iter_vars.append(pred_mean_var[:, 1:].cpu().flatten())  # Variance Column (2nd)

            preds_mean = torch.stack(iter_means)
            preds_var = torch.stack(iter_vars)

        preds_mean = preds_mean.mean(dim=0)
        preds_var = preds_var.mean(dim=0)
        preds_std = torch.sqrt(preds_var)

        if self.undo_boneage_rescale:
            preds_mean = undo_boneage_rescale(preds_mean)
            preds_var = undo_boneage_rescale(preds_var)
            preds_std = undo_boneage_rescale(preds_std)

        metrics = {
            ADT_STAT_PREDS_VAR: preds_var,
        }
        return preds_mean, preds_std, metrics

    def evaluate_dataset(
        self, dataloader: DataLoader
    ) -> tuple[Any, Tensor, Tensor, Tensor, Tensor, Optional[dict[str, Any]]]:
        self.eval()
        self.cuda()
        return uam_evaluate_dataset_default(self, self.device, dataloader)
