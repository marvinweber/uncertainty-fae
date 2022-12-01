from typing import Any, Optional

import torch
import tqdm
from torch import Tensor
from torch.utils.data import DataLoader

from rsna_boneage.data import undo_boneage_rescale
from rsna_boneage.litmodel.dropout import (LitRSNABoneageMCDropout,
                                           LitRSNABoneageVarianceNetMCDropout)
from uncertainty_fae.model import EvaluationMetrics, TrainLoadMixin, UncertaintyAwareModel


class LitRSNABoneageDeepEnsemble(UncertaintyAwareModel, TrainLoadMixin):

    BASE_MODEL_CLASS = LitRSNABoneageMCDropout  # We won't use the dropout during inference

    def __init__(
        self,
        *args,
        base_model_checkpoints: Optional[list[str]] = None,
        undo_boneage_rescale: Optional[bool] = False,
        **kwargs,
    ) -> None:
        self.base_model_checkpoints = base_model_checkpoints
        self.base_model_args = args
        self.base_model_kwargs = kwargs

        self.undo_boneage_rescale = undo_boneage_rescale

        super().__init__()

    @classmethod
    def load_model_from_disk(cls, checkpoint_path, base_model_checkpoints, **kwargs):
        assert isinstance(base_model_checkpoints, list)
        model = cls(base_model_checkpoints=base_model_checkpoints, **kwargs)
        return model

    def evaluate_dataset(
        self, dataloader: DataLoader
    ) -> tuple[Any, Tensor, Tensor, Tensor, Tensor, EvaluationMetrics]:
        assert self.base_model_checkpoints and len(self.base_model_checkpoints) > 0

        n_predictions = []  # list of tensors, each tensor is a prediction set for all samples
        targets = []

        for ckpt_path in tqdm.tqdm(self.base_model_checkpoints, desc='DE Model Progress'):
            model = self.BASE_MODEL_CLASS.load_model_from_disk(
                ckpt_path, *self.base_model_args, **self.base_model_kwargs)
            assert isinstance(model, self.BASE_MODEL_CLASS)

            model.eval()
            model.cuda()

            with torch.no_grad():
                model_predictions = []  # prediction for every sample
                data_iterator = tqdm.tqdm(
                    dataloader, desc=f'Prediction Progress', total=len(dataloader), leave=False)

                for input, target in data_iterator:
                    # fill targets on first iteration
                    if len(targets) < len(dataloader):
                        targets.append(target)

                    input = input.cuda()
                    pred_y = model.forward(input)
                    model_predictions.append(pred_y.cpu().flatten())

                model_predictions = torch.cat(model_predictions)
                n_predictions.append(model_predictions)

        targets = torch.cat(targets)
        n_predictions = torch.stack(n_predictions)
        preds_mean = n_predictions.mean(dim=0)
        preds_std = n_predictions.std(dim=0)

        if self.undo_boneage_rescale:
            n_predictions = undo_boneage_rescale(n_predictions)
            preds_mean = undo_boneage_rescale(preds_mean)
            preds_std = undo_boneage_rescale(preds_std)

        preds_abs_errors = torch.abs((preds_mean - targets))
        mae = torch.mean(preds_abs_errors)
        distinct_model_maes = [torch.mean(torch.abs(preds - targets)) for preds in n_predictions]

        eval_metrics = EvaluationMetrics(
            preds_distinct=[n_predictions[:, i:i+1].flatten() for i in range(len(preds_mean))],
            mean_uncertainty=preds_std.mean(),
            distinct_model_errors=distinct_model_maes,
        )
        return mae, preds_mean, targets, preds_abs_errors, preds_std, eval_metrics

    @classmethod
    def train_model(cls, *args, **kwargs):
        raise NotImplementedError('Use base model class for training!')


class LitRSNABoneageVarianceNetDeepEnsemble(LitRSNABoneageDeepEnsemble):

    BASE_MODEL_CLASS = LitRSNABoneageVarianceNetMCDropout  # We won't use the dropout for inference

    def __init__(
        self,
        *args,
        base_model_checkpoints: Optional[list[str]] = None,
        undo_boneage_rescale: Optional[bool] = False,
        **kwargs
    ) -> None:
        super().__init__(
            *args,
            base_model_checkpoints=base_model_checkpoints,
            undo_boneage_rescale=undo_boneage_rescale,
            **kwargs,
        )

    def evaluate_dataset(
        self, dataloader: DataLoader
    ) -> tuple[Any, Tensor, Tensor, Tensor, Tensor, EvaluationMetrics]:
        assert self.base_model_checkpoints and len(self.base_model_checkpoints) > 0

        n_predictions = []  # list of tensors, each tensor is a prediction set for all samples
        n_variances = []  # list of tensors, each tensor is a variance set for all samples
        targets = []

        for ckpt_path in tqdm.tqdm(self.base_model_checkpoints, desc='DE Model Progress'):
            model = self.BASE_MODEL_CLASS.load_model_from_disk(
                ckpt_path, *self.base_model_args, **self.base_model_kwargs)
            assert isinstance(model, self.BASE_MODEL_CLASS)

            model.eval()
            model.cuda()

            with torch.no_grad():
                model_predictions = []  # prediction of every sample for current model
                model_variances = []  # variances of every sample for current model
                data_iterator = tqdm.tqdm(
                    dataloader, desc=f'Prediction Progress', total=len(dataloader), leave=False)

                for input, target in data_iterator:
                    # fill targets on first iteration
                    if len(targets) < len(dataloader):
                        targets.append(target)

                    input = input.cuda()
                    pred_mean_var = model.forward(input)
                    model_predictions.append(pred_mean_var[:, :1].cpu().flatten())
                    model_variances.append(pred_mean_var[:, 1:].cpu().flatten())

                model_predictions = torch.cat(model_predictions)
                model_variances = torch.cat(model_variances)
                n_predictions.append(model_predictions)
                n_variances.append(model_variances)

        targets = torch.cat(targets)
        n_predictions = torch.stack(n_predictions)
        n_variances = torch.stack(n_variances)
        preds_mean = n_predictions.mean(dim=0)
        preds_var = n_variances.mean(dim=0)
        preds_std = torch.sqrt(preds_var)

        if self.undo_boneage_rescale:
            n_predictions = undo_boneage_rescale(n_predictions)
            preds_mean = undo_boneage_rescale(preds_mean)
            preds_std = undo_boneage_rescale(preds_std)

        preds_abs_errors = torch.abs((preds_mean - targets))
        mae = torch.mean(preds_abs_errors)
        distinct_model_maes = [torch.mean(torch.abs(preds - targets)) for preds in n_predictions]

        eval_metrics = EvaluationMetrics(
            preds_distinct=[n_predictions[:, i:i+1].flatten() for i in range(len(preds_mean))],
            mean_uncertainty=preds_std.mean(),
            distinct_model_errors=distinct_model_maes,
        )
        return mae, preds_mean, targets, preds_abs_errors, preds_std, eval_metrics
