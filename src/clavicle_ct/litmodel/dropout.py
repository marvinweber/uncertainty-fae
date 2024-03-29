from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from clavicle_ct.data import undo_clavicle_age_rescale
from clavicle_ct.litmodel.base import LitClavicle, LitClavicleVarianceNet
from uncertainty_fae.model import (
    EvaluationMetrics,
    ForwardMetrics,
    UncertaintyAwareModel,
    uam_evaluate_dataset_default,
)
from uncertainty_fae.util import dropout_train


class LitClavicleMCDropout(UncertaintyAwareModel, LitClavicle):
    def __init__(self, *args, n_samples: int = 100, **kwargs) -> None:
        self.n_samples = n_samples
        super().__init__(*args, **kwargs)

    def forward_with_uncertainty(self, batch) -> tuple[Tensor, Tensor, ForwardMetrics]:
        # Enable Dropout Layers in Network for MC
        self.apply(dropout_train)

        with torch.no_grad():
            preds = [self.forward(batch).cpu().flatten() for _ in range(self.n_samples)]
            preds = torch.stack(preds)

        preds_mean = preds.mean(dim=0)
        preds_std = preds.std(dim=0)

        if self.undo_boneage_rescale:
            preds = undo_clavicle_age_rescale(preds)
            preds_mean = undo_clavicle_age_rescale(preds_mean)
            preds_std = undo_clavicle_age_rescale(preds_std, with_shift=False)

        forward_metrics = ForwardMetrics(
            preds_distinct=[preds[:, i : i + 1].flatten() for i in range(len(preds_mean))],
        )
        return preds_mean, preds_std, forward_metrics

    def evaluate_dataset(
        self, dataloader: DataLoader
    ) -> tuple[Any, Tensor, Tensor, Tensor, Tensor, EvaluationMetrics]:
        self.eval()
        self.cuda()
        return uam_evaluate_dataset_default(self, self.device, dataloader)


class LitClavicleVarianceNetMCDropout(LitClavicleVarianceNet):
    def __init__(self, *args, n_samples: int = 100, **kwargs) -> None:
        self.n_samples = n_samples
        super().__init__(*args, **kwargs)

    def forward_with_uncertainty(self, batch) -> tuple[Tensor, Tensor, ForwardMetrics]:
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
            preds_mean = undo_clavicle_age_rescale(preds_mean)
            preds_std = undo_clavicle_age_rescale(preds_std, with_shift=False)

        return preds_mean, preds_std, ForwardMetrics()

    def evaluate_dataset(
        self, dataloader: DataLoader
    ) -> tuple[Any, Tensor, Tensor, Tensor, Tensor, EvaluationMetrics]:
        self.eval()
        self.cuda()
        return uam_evaluate_dataset_default(self, self.device, dataloader)
