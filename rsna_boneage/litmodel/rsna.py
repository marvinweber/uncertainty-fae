
import gzip
import logging
import os
from typing import Any, Tuple

import dill
import torch
from laplace import BaseLaplace, KronLLLaplace, Laplace
from pytorch_lightning import LightningDataModule
from torch import vstack

from rsna_boneage.litmodel import LitRSNABoneage
from rsna_boneage.net.inception import RSNABoneageInceptionNetWithGender
from rsna_boneage.net.resnet import RSNABoneageResNetWithGender
from uncertainty.model import TrainLoadMixin, UncertaintyAwareModel
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


class LitRSNABoneageLaplace(UncertaintyAwareModel, TrainLoadMixin):

    BASE_MODEL_CLASS = LitRSNABoneage
    """Class type of the base model trained before Laplace approximation."""

    def __init__(self, *args, n_samples: int = 100, **kwargs):
        super().__init__()

        self.n_samples = n_samples
        self.base_model = self.BASE_MODEL_CLASS(*args, **kwargs)
        self.la_model = None

    def forward(self, x: Any) -> Any:        
        if self.la_model and isinstance(self.la_model, BaseLaplace):
            # TODO: use la model for prediction and throw away uncertainty
            pass
        elif self.base_model and isinstance(self.base_model, self.BASE_MODEL_CLASS):
            # TODO: use base model for "normal" prediction
            pass
        else:
            raise ValueError('Neither base_model, nor la_model are available. No Forward possible!')

        raise NotImplementedError('Not yet ready!')

    def forward_with_uncertainty(self, input) -> Tuple[torch.Tensor, Any]:
        assert isinstance(self.la_model, BaseLaplace), \
            'Loaded model has not yet been last-layer laplace approximated (no la_model available)!'

        predictions = self.la_model(input, pred_type='nn', link_approx='mc',
                                    n_samples=self.n_samples)
        return predictions

    def evaluate_dataset(self, dataloader):
        raise NotImplementedError('Not yet implemented')

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

        return model

    @classmethod
    def train_model(cls, log_dir: str, datamodule: LightningDataModule,
                    model: 'LitRSNABoneageLaplace', train_config: TrainConfig,
                    is_resume: bool = False) -> TrainResult:
        assert isinstance(model.base_model, model.BASE_MODEL_CLASS)
        logger = logging.getLogger('LAPLACE-LITMODEL')
        logger.info('Starting Base-Model Training...')

        # train the base model
        train_result_base_model = model.base_model.__class__.train_model(
            log_dir, datamodule, model.base_model, train_config, is_resume)

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


        return TrainResult(interrupted=False)
