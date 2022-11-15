import gzip
import logging
import os
from copy import deepcopy
from typing import Any, Optional, Union

import torch
import tqdm
from pytorch_lightning import Callback, LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import trange

import swa_gaussian.utils as swag_utils
from rsna_boneage.data import undo_boneage_rescale
from rsna_boneage.litmodel.base import LitRSNABoneage, LitRSNABoneageVarianceNet
from swa_gaussian.pl_callback.swag_callback import SWAGaussianCallback
from swa_gaussian.posteriors.swag import SWAG
from uncertainty_fae.model import ADT_STAT_MEAN_UNCERTAINTY, TrainLoadMixin, UncertaintyAwareModel
from uncertainty_fae.swag import SwagEvalCallback
from uncertainty_fae.util import TrainConfig, TrainResult

logger = logging.getLogger(__name__)


class LitRSNABoneageSWAG(UncertaintyAwareModel, TrainLoadMixin):

    BASE_MODEL_CLASS = LitRSNABoneage

    def __init__(
        self,
        *args,
        lr: float = 0.0005,
        n_samples: int = 30,
        optim_type: str = 'sgd',
        swa_lrs: Union[float, list[float]] = 0.0003,
        swa_start_epoch: int = 31,
        swa_annealing_epochs: int = 10,
        swa_annealing_strategy: str = 'linear',
        swag_max_num_models: int = 20,
        swag_sample_scale: float = 1.0,
        swag_model: Optional[SWAG] = None,
        base_model_checkpoint_pth: Optional[str] = None,
        undo_boneage_rescale: Optional[bool] = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.n_samples = n_samples
        self.swag_sample_scale = swag_sample_scale

        self.undo_boneage_rescale = undo_boneage_rescale

        # SWA(G) Training Configuration
        self.swa_lrs = swa_lrs
        self.swa_start_epoch = swa_start_epoch
        self.swa_annealing_epochs = swa_annealing_epochs
        self.swa_annealing_strategy = swa_annealing_strategy
        self.swag_with_cov = True  # not configurable for now

        # Either load given base model checkpoint, or create "fresh" one
        if base_model_checkpoint_pth:
            self.base_model = self.BASE_MODEL_CLASS.load_from_checkpoint(
                base_model_checkpoint_pth, lr=lr, optim_type=optim_type, **kwargs)
        else:
            self.base_model = self.BASE_MODEL_CLASS(*args, lr=lr, optim_type=optim_type, **kwargs)

        # Either use given SWAG model, or create one from base model
        if swag_model is not None and isinstance(swag_model, SWAG):
            self.swag_model = swag_model
        else:
            with self.base_model._prevent_trainer_and_dataloaders_deepcopy():
                self.swag_model = SWAG(
                    deepcopy(self.base_model),
                    no_cov_mat=(not self.swag_with_cov),
                    max_num_models=swag_max_num_models,
                )

    def evaluate_dataset(
        self,
        dataloader: DataLoader
    ) -> tuple[Any, Tensor, Tensor, Tensor, Tensor, Optional[dict[str, Any]]]:
        assert self.train_dataloader and isinstance(self.train_dataloader, DataLoader), \
            'SWAG requires the train dataloader to be set (c.f. `set_dataloaders(...)`!'

        # We currently need the model to be on cuda
        self.swag_model.cuda()

        n_predictions = []  # list of n tensors, each tensor is a prediction set for all samples
        targets = []

        n_iterator = trange(self.n_samples, desc='n_samples (per item)')
        for n in n_iterator:
            # Sample and update batch norm layers
            self.swag_model.sample(scale=self.swag_sample_scale, cov=self.swag_with_cov, seed=n)
            swag_utils.bn_update_2(self.train_dataloader, self.swag_model, cuda=True)
            n_iterator.refresh()  # ensure global progess is shown

            # Ensure eval mode for predictions
            self.swag_model.eval()

            # Make predictions
            iter_preds = []  # prediction for every sample
            data_iterator = tqdm.tqdm(dataloader, desc=f'predictions for iteration n={n}',
                                      total=len(dataloader), leave=False)
            for input, target in data_iterator:
                # fill targets on first iteration
                if n == 0:
                    targets.append(target)

                input = input.cuda()
                pred_y = self.swag_model(input)
                iter_preds.append(pred_y.cpu().flatten())

            iter_preds = torch.cat(iter_preds)
            n_predictions.append(iter_preds)

        targets = torch.cat(targets)
        preds_mean = torch.stack(n_predictions).mean(dim=0)
        preds_var = torch.stack(n_predictions).var(dim=0)
        preds_std = torch.stack(n_predictions).std(dim=0)

        if self.undo_boneage_rescale:
            preds_mean = undo_boneage_rescale(preds_mean)
            preds_var = undo_boneage_rescale(preds_var)
            preds_std = undo_boneage_rescale(preds_std)

        preds_abs_errors = torch.abs((preds_mean - targets))
        mae = torch.mean(preds_abs_errors)

        metrics = {
            ADT_STAT_MEAN_UNCERTAINTY: preds_std.mean(),
            'mae': mae,
        }
        return mae, preds_mean, targets, preds_abs_errors, preds_std, metrics

    @classmethod
    def load_model_from_disk(cls, checkpoint_path: str, base_model_checkpoint_pth: str = None,
                             **kwargs) -> 'LitRSNABoneageSWAG':
        logger.info('Loading SWAG Model from file...')
        with gzip.open(checkpoint_path, 'rb') as file:
            swag_model = torch.load(file)
        assert isinstance(swag_model, SWAG), 'Given checkpoint is not of a SWAG model!'
        model = cls(swag_model=swag_model, base_model_checkpoint_pth=base_model_checkpoint_pth,
                    **kwargs)
        logger.info('%s loaded', cls.__name__)
        return model

    @classmethod
    def train_model(cls, log_dir: str, datamodule: LightningDataModule, model: 'LitRSNABoneageSWAG',
                    train_config: TrainConfig, is_resume: bool = False,
                    callbacks: Optional[list[Callback]] = None) -> TrainResult:
        assert isinstance(model.base_model, model.BASE_MODEL_CLASS)

        # Ensure no early stopping is done
        if train_config.early_stopping_patience < train_config.max_epochs:
            logger.warning(
                f'Configured early stopping patience {train_config.early_stopping_patience} will '
                'be ignored (i.e., early stopping is disabled now)!')
            train_config.early_stopping_patience = train_config.max_epochs

        # SWA(G) Evaluation
        # We want to evaluate the performance of the SWA model (not the SWAG uncertainty and also
        # not the SWAG predictions (due to too large computational overhead)).
        # Thus we set n_samples=1 and swag_sample_scale=0. See
        # https://github.com/wjmaddox/swa_gaussian/blob/master/experiments/train/run_swag.py#L322..L329
        # for details on original SWAG implementation and their "in training" evaluation.
        orig_params = {
            'n_samples': model.n_samples,
            'swag_sample_scale': model.swag_sample_scale,
        }
        model.n_samples = 1
        model.swag_sample_scale = 0

        datamodule.setup('fit')
        model.set_dataloaders(train_dataloader=datamodule.train_dataloader())

        # SWAG Callbacks
        swag_callback = SWAGaussianCallback(
            model.swag_model,
            model.swa_lrs,
            swa_start_epoch=model.swa_start_epoch,
            annealing_epochs=model.swa_annealing_epochs,
            annealing_strategy=model.swa_annealing_strategy,
        )
        swag_eval_callback = SwagEvalCallback(model, model.swa_start_epoch, ['mae'])
        train_callbacks = [*(callbacks if callbacks else []), swag_callback, swag_eval_callback]

        logger.info('Starting Base-Model Training...')
        train_result_base_model = model.base_model.__class__.train_model(
            log_dir=log_dir, datamodule=datamodule, model=model.base_model,
            train_config=train_config, is_resume=is_resume, callbacks=train_callbacks)

        if train_result_base_model.interrupted:
            # we got interrupted and should return here
            logger.info('Base model training was interrupted, returning without saving.')
            return train_result_base_model

        # Restore model parameters
        for param, val in orig_params.items():
            setattr(model, param, val)

        # Store SWAG model
        swag_model_fpath = os.path.join(log_dir, 'swag_model.gz')
        logger.info('Dumping the swag model to: %s', swag_model_fpath)
        with gzip.open(swag_model_fpath, 'wb') as file:
            torch.save(model.swag_model, file)

        additional_info = {
            'base_model_best_model_path': train_result_base_model.best_model_path
        }
        return TrainResult(interrupted=False, best_model_path=swag_model_fpath,
                           additional_info=additional_info)


class LitRSNABoneageVarianceNetSWAG(LitRSNABoneageSWAG):

    BASE_MODEL_CLASS = LitRSNABoneageVarianceNet

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def evaluate_dataset(
        self,
        dataloader: DataLoader
    ) -> tuple[Any, Tensor, Tensor, Tensor, Tensor, Optional[dict[str, Any]]]:
        assert self.train_dataloader and isinstance(self.train_dataloader, DataLoader), \
            'SWAG requires the train dataloader to be set (c.f. `set_dataloaders(...)`!'

        # We currently need the model to be on cuda
        self.swag_model.cuda()

        n_predictions = []  # list of n tensors, each tensor is a prediction set for all samples
        n_variances = []  # list of n tensors, each is a predicted variance set for all samples
        targets = []

        n_iterator = trange(self.n_samples, desc='n_samples (per item)')
        for n in n_iterator:
            # Sample and update batch norm layers
            self.swag_model.sample(scale=self.swag_sample_scale, cov=self.swag_with_cov, seed=n)
            swag_utils.bn_update_2(self.train_dataloader, self.swag_model, cuda=True)
            n_iterator.refresh()  # ensure global progess is shown

            # Ensure eval mode for predictions
            self.swag_model.eval()

            # Make predictions
            iter_preds = []  # prediction for every sample
            iter_vars = []  # "predicted" variance for every sample
            data_iterator = tqdm.tqdm(dataloader, desc=f'predictions for iteration n={n}',
                                      total=len(dataloader), leave=False)
            for input, target in data_iterator:
                # fill targets on first iteration
                if n == 0:
                    targets.append(target)

                input = input.cuda()
                pred_mean_var: Tensor = self.swag_model(input)
                iter_preds.append(pred_mean_var[:, :1].cpu().flatten())
                iter_vars.append(pred_mean_var[:, 1:].cpu().flatten())

            iter_preds = torch.cat(iter_preds)
            iter_vars = torch.cat(iter_vars)
            n_predictions.append(iter_preds)
            n_variances.append(iter_vars)

        targets = torch.cat(targets)
        preds_mean = torch.stack(n_predictions).mean(dim=0)
        preds_var = torch.stack(n_variances).mean(dim=0)
        preds_std = torch.sqrt(preds_var)

        if self.undo_boneage_rescale:
            preds_mean = undo_boneage_rescale(preds_mean)
            preds_var = undo_boneage_rescale(preds_var)
            preds_std = undo_boneage_rescale(preds_std)

        preds_abs_errors = torch.abs((preds_mean - targets))
        mae = torch.mean(preds_abs_errors)

        metrics = {
            ADT_STAT_MEAN_UNCERTAINTY: preds_std.mean(),
            'mae': mae,
        }
        return mae, preds_mean, targets, preds_abs_errors, preds_std, metrics
