import gzip
import logging
import os
from copy import deepcopy
from typing import Any, Optional, Tuple, Union

import dill
import torch
import tqdm
from laplace import BaseLaplace, KronLLLaplace, Laplace
from laplace.utils import FeatureExtractor
from pytorch_lightning import Callback, LightningDataModule
from torch import Tensor, vstack
from torch.utils.data import DataLoader
from tqdm import trange
from rsna_boneage.data import undo_boneage_rescale

import swa_gaussian.utils as swag_utils
from rsna_boneage.litmodel import LitRSNABoneage
from rsna_boneage.net.inception import RSNABoneageInceptionNetWithGender
from rsna_boneage.net.resnet import RSNABoneageResNetWithGender
from swa_gaussian.pl_callback.swag_callback import SWAGaussianCallback
from swa_gaussian.posteriors.swag import SWAG
from uncertainty.model import TrainLoadMixin, UncertaintyAwareModel
from uncertainty.swag import SwagEvalCallback
from util.training import TrainConfig, TrainResult


class LitRSNABoneageMCDropout(UncertaintyAwareModel, LitRSNABoneage):

    def __init__(self, *args, mc_iterations: int = 100, **kwargs):
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


class LitRSNABoneageLaplace(UncertaintyAwareModel, TrainLoadMixin):

    BASE_MODEL_CLASS = LitRSNABoneage
    """Class type of the base model trained before Laplace approximation."""

    def __init__(self, *args, n_samples: int = 100, **kwargs):
        super().__init__()

        self.n_samples = n_samples
        self.base_model = self.BASE_MODEL_CLASS(*args, **kwargs)
        self.la_model: BaseLaplace = None

    @torch.no_grad()
    def forward_without_uncertainty(self, x: torch.Tensor):
        if self.la_model and isinstance(self.la_model, BaseLaplace):
            # TODO: use la model for prediction and throw away uncertainty
            raise NotImplementedError('Not yet ready!')
        elif self.base_model and isinstance(self.base_model, self.BASE_MODEL_CLASS):
            if torch.cuda.is_available():
                self.base_model.cuda()
            self.base_model.eval()
            return self.base_model(x.to(self.base_model.device))
        else:
            raise ValueError('Neither base_model, nor la_model are available. No Forward possible!')

    @torch.no_grad()
    def forward_with_uncertainty(self, input) -> Tuple[torch.Tensor, Any]:
        assert isinstance(self.la_model, BaseLaplace), \
            'Loaded model has not yet been last-layer laplace approximated (no la_model available)!'

        self.la_model.model.eval()
        predictions = self.la_model(input, pred_type='nn', link_approx='mc',
                                    n_samples=self.n_samples)
        return predictions

    def evaluate_dataset(
        self,
        dataloader: DataLoader
    ) -> tuple[Any, Tensor, Tensor, Tensor, Tensor, dict[Any, dict], dict[str, Any]]:
        raise NotImplementedError('Not yet implemented')  # TODO: implement

    @classmethod
    def load_model_from_disk(cls, checkpoint_path: str, base_model_checkpoint_pth: str = None,
                             **kwargs) -> 'LitRSNABoneageLaplace':
        with gzip.open(checkpoint_path, 'rb') as file:
            model = dill.load(file)

        assert isinstance(model, LitRSNABoneageLaplace), 'Loaded model is not of correct instance!'
        assert isinstance(model.la_model, BaseLaplace), \
            ('Loaded model has not yet been last-layer laplace approximated (no la_model found in '
             'the serialized file)!')

        if base_model_checkpoint_pth is not None:
            if not os.path.isfile(base_model_checkpoint_pth):
                raise ValueError(
                    f'Given base model checkpoint path is not valid: {base_model_checkpoint_pth}')
            model.base_model = cls.BASE_MODEL_CLASS.load_from_checkpoint(
                base_model_checkpoint_pth, **kwargs)

        if torch.cuda.is_available():
            model.la_model.model.cuda()
            if isinstance(model.la_model.model, FeatureExtractor):
                model.la_model.model.cuda()
                model.la_model._device = model.la_model.model.model.device

        return model

    @classmethod
    def train_model(cls, log_dir: str, datamodule: LightningDataModule,
                    model: 'LitRSNABoneageLaplace', train_config: TrainConfig,
                    is_resume: bool = False, callbacks: Optional[list[Callback]] = None) -> TrainResult:
        assert isinstance(model.base_model, model.BASE_MODEL_CLASS)
        logger = logging.getLogger('LAPLACE-LITMODEL')
        logger.info('Starting Base-Model Training...')

        # train the base model
        train_result_base_model = model.base_model.__class__.train_model(
            log_dir=log_dir, datamodule=datamodule, model=model.base_model,
            train_config=train_config, is_resume=is_resume, callbacks=callbacks)

        if train_result_base_model.interrupted:
            # we got interrupted and should not continue with laplace
            logger.info('Base model training was interrupted, returning without LA approximation.')
            return train_result_base_model

        logger.debug('Loading best model state into base model...')
        if (not train_result_base_model.best_model_path
                or not os.path.isfile(train_result_base_model.best_model_path)):
            logger.error(
                'Base model training did not return valid best_model_path (%s)! Continue '
                'without loading best model weights!', train_result_base_model.best_model_path)
        else:
            checkpoint = torch.load(train_result_base_model.best_model_path)
            assert 'state_dict' in checkpoint, \
                'No "model_state_dict" in the base model checkpoint!'
            model.base_model.load_state_dict(checkpoint['state_dict'])

        logger.info('Laplace Training: Base model training finished. Starting LA Fit...')

        # Laplace fit is performed on the device the model is located on -> use GPU if available
        if torch.cuda.is_available():
            model.base_model.cuda()
        else:
            logger.warning('Cuda not available; LA fit on CPU may take very long!')

        la: KronLLLaplace = Laplace(model.base_model, 'regression', 'last_layer', 'kron')
        la.fit(datamodule.train_dataloader())
        model.la_model = la

        logger.info('LA fit done; dumping model to file...')

        # store model in eval mode
        model.la_model.model.eval()

        # base model can be restored from checkpoint file -> don't serialize it twice
        del model.base_model
        model.base_model = None
        laplace_model_fpath = os.path.join(log_dir, 'laplace_model.gz')
        with gzip.open(laplace_model_fpath, 'wb') as file:
            dill.dump(model, file)
        logger.debug('LA file dumping done!')

        additional_info = {
            'base_model_best_model_path': train_result_base_model.best_model_path
        }
        return TrainResult(interrupted=False, best_model_path=laplace_model_fpath,
                           additional_info=additional_info)


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
    ):
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
    ) -> tuple[Any, Tensor, Tensor, Tensor, Tensor, dict[Any, dict], dict[str, Any]]:
        assert self.train_dataloader and isinstance(self.train_dataloader, DataLoader), \
            'SWAG requires the train dataloader to be set (c.f. `set_dataloaders(...)`!'

        # We currently need the model to be on cuda
        self.swag_model.cuda()

        n_predictions = []
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
            iter_preds = []
            data_iterator = tqdm.tqdm(dataloader, desc=f'predictions for iteration n={n}',
                                      total=len(dataloader), leave=False)
            for input, target in data_iterator:
                # fill targets on first iteration
                if n == 0:
                    targets.append(target)

                input = input.cuda()
                pred_y = self.swag_model(input)
                iter_preds.append(pred_y.cpu())

            iter_preds = torch.cat(iter_preds).flatten()
            n_predictions.append(iter_preds)

        targets = torch.cat(targets)
        mean_predictions = torch.stack(n_predictions).mean(dim=0)
        var_predictions = torch.stack(n_predictions).var(dim=0)
        std_predictions = torch.stack(n_predictions).std(dim=0)

        if self.undo_boneage_rescale:
            targets = undo_boneage_rescale(targets)
            mean_predictions = undo_boneage_rescale(mean_predictions)
            var_predictions = undo_boneage_rescale(var_predictions)
            std_predictions = undo_boneage_rescale(std_predictions)

        abs_errors_predictions = torch.abs((mean_predictions - targets))
        mae = torch.mean(abs_errors_predictions)

        # TODO: extend metrics
        all_metrics = {
            'mae': mae,
            'mean_uncertainty': std_predictions.mean()
        }
        return (mae, mean_predictions, targets, abs_errors_predictions, std_predictions, [],
                all_metrics)

    @classmethod
    def load_model_from_disk(cls, checkpoint_path: str, base_model_checkpoint_pth: str = None,
                             **kwargs) -> 'LitRSNABoneageSWAG':
        with gzip.open(checkpoint_path, 'rb') as file:
            swag_model = torch.load(file)
        assert isinstance(swag_model, SWAG), 'Given checkpoint is not of a SWAG model!'
        model = cls(swag_model=swag_model, base_model_checkpoint_pth=base_model_checkpoint_pth,
                    **kwargs)
        return model

    @classmethod
    def train_model(cls, log_dir: str, datamodule: LightningDataModule, model: 'LitRSNABoneageSWAG',
                    train_config: TrainConfig, is_resume: bool = False,
                    callbacks: Optional[list[Callback]] = None) -> TrainResult:
        assert isinstance(model.base_model, model.BASE_MODEL_CLASS)
        logger = logging.getLogger('SWAG-LITMODEL')

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
