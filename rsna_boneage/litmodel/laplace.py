import gzip
import logging
import os
from typing import Any, Optional

import dill
import torch
from laplace import BaseLaplace, KronLLLaplace, Laplace
from laplace.utils import FeatureExtractor
from pytorch_lightning import Callback, LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader

from rsna_boneage.data import undo_boneage_rescale
from uncertainty.model import (ADT_STAT_PREDS_VAR, TrainLoadMixin, UncertaintyAwareModel,
                               uam_evaluate_dataset_default)
from util.training import TrainConfig, TrainResult

logger = logging.getLogger(__name__)


class LitRSNABoneageLaplace(UncertaintyAwareModel, TrainLoadMixin):

    BASE_MODEL_CLASS = LitRSNABoneage
    """Class type of the base model trained before Laplace approximation."""

    def __init__(
        self,
        *args,
        n_samples: int = 100,
        undo_boneage_rescale: Optional[bool] = False,
        **kwargs
    ) -> None:
        super().__init__()

        self.n_samples = n_samples
        self.undo_boneage_rescale = undo_boneage_rescale
        self.base_model = self.BASE_MODEL_CLASS(*args, **kwargs)
        self.la_model: BaseLaplace = None

        self.laplace_backend_loss_fn_decorator = laplace_backend_loss_fn_decorator_standard
        """Decorator/ Wrapper for Laplace Backend Loss function; see train method for details."""

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
    def forward_with_uncertainty(self, input) -> tuple[Tensor, Tensor, Optional[dict[str, Any]]]:
        assert isinstance(self.la_model, BaseLaplace), \
            'Loaded model has not yet been last-layer laplace approximated (no la_model available)!'

        self.la_model.model.eval()
        preds_mean, preds_var = self.la_model(
            input,
            pred_type='nn',
            link_approx='mc',
            n_samples=self.n_samples
        )
        preds_mean = preds_mean.cpu().flatten()
        preds_var = preds_var.cpu().flatten()
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
        self,
        dataloader: DataLoader
    ) -> tuple[Any, Tensor, Tensor, Tensor, Tensor, dict[Any, dict], dict[str, Any]]:
        return uam_evaluate_dataset_default(self, 'cuda', dataloader)

    @classmethod
    def load_model_from_disk(
        cls,
        checkpoint_path: str,
        base_model_checkpoint_pth: str = None,
        n_samples: int = 100,
        undo_boneage_rescale: Optional[bool] = False,
        **kwargs,
    ) -> 'LitRSNABoneageLaplace':
        logger.info('Loading Laplace Model from file...')
        with gzip.open(checkpoint_path, 'rb') as file:
            model = dill.load(file)

        assert isinstance(model, LitRSNABoneageLaplace), 'Loaded model is not of correct instance!'
        assert isinstance(model.la_model, BaseLaplace), \
            ('Loaded model has not yet been last-layer laplace approximated (no la_model found in '
             'the serialized file)!')
        # We must explicitly update the model parameters (the ones from training, stored in the
        # dumped model, could be different from the ones used when loading the model again).
        model.n_samples = n_samples
        model.undo_boneage_rescale = undo_boneage_rescale

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

        logger.info('%s loaded', cls.__name__)
        return model

    @classmethod
    def train_model(cls, log_dir: str, datamodule: LightningDataModule,
                    model: 'LitRSNABoneageLaplace', train_config: TrainConfig,
                    is_resume: bool = False, callbacks: Optional[list[Callback]] = None) -> TrainResult:
        assert isinstance(model.base_model, model.BASE_MODEL_CLASS)
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

        # We need to adjust the loss function used by the laplace backend, because per default MSE
        # is used and fed with a column prediction vector and a row target value which results in
        # too high loss values.
        # Solution: wrap the loss function an unsqueeze the targets. For this, we need to access the
        # backend, which requires the last_layer to be known.
        X, _ = next(iter(datamodule.train_dataloader()))
        la.model.find_last_layer(X.cuda() if torch.cuda.is_available() else X)
        la.backend.lossfunc = model.laplace_backend_loss_fn_decorator(la.backend.lossfunc)
        # Workaround, because the Laplace (KronLLLaplace::fit()) fit-method only sets some other
        # parameters (e.g., n_params), when the last_layer is not set already.
        la.model.last_layer = None

        # Perform Laplace Last-Layer Approximation and Optimize Prior Precision
        la.fit(datamodule.train_dataloader())
        logger.info('Starting prior precision optimization...')
        la.optimize_prior_precision(pred_type='nn')

        logger.info('LA Fit and Optimization done; dumping model to file...')
        # store model in eval mode
        model.la_model = la
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


def laplace_backend_loss_fn_decorator_standard(loss_fn):
    def inner(x: Tensor, y: Tensor):
        y = y.unsqueeze(1)
        return loss_fn(x, y)
    return inner
