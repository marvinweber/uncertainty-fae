
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
from uncertainty.model import UncertaintyAwareModel
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


class LitRSNABoneageLaplace(UncertaintyAwareModel, LitRSNABoneage):

    def __init__(self, *args, n_samples: int = 30, **kwargs):
        # this will initialize net and so on on this class well -> we don't need this as we only
        # use base_model plus la model later, but for consistency reasons super is still called
        super().__init__(*args, **kwargs)
        self.n_samples = n_samples
        self.base_model = LitRSNABoneage(*args, **kwargs)
        self.la_model = None

    def forward_with_uncertainty(self, input) -> Tuple[torch.Tensor, Any]:
        assert isinstance(self.la_model, BaseLaplace), \
            'Loaded model has not yet been last-layer laplace approximated!'

        # TODO: verify correctness
        predictions = self.la_model(input)
        return predictions

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, base_model_checkpoint_pth: str = None,
                             **kwargs) -> 'LitRSNABoneageLaplace':
        with gzip.open(checkpoint_path, 'rb') as file:
            model = dill.load(file)

        assert isinstance(model, LitRSNABoneageLaplace), 'Loaded model is not of correct instance!'
        assert isinstance(model.la_model, BaseLaplace), \
            'Loaded model has yet been last-layer laplace approximated!'
        
        if base_model_checkpoint_pth is not None:
            if not os.path.isfile(base_model_checkpoint_pth):
                raise ValueError(
                    f'Given base model checkpoint path is not valid: {base_model_checkpoint_pth}')
            # TODO: load base model as well

        return model

    @classmethod
    def train_model(cls, log_dir: str, datamodule: LightningDataModule,
                    model: 'LitRSNABoneageLaplace', train_config: TrainConfig,
                    is_resume: bool = False) -> TrainResult:
        # train the base model
        train_result = super().train_model(
            log_dir, datamodule, model.base_model, train_config, is_resume)

        # TODO: load state dict of best epoch (not last) into base model

        if train_result.interrupted:
            # we got interrupted and should not continue with laplace
            return train_result

        logger = logging.getLogger('LAPLACE-LITMODEL')
        logger.info('Laplace Training: Base model training finished. Starting LA Fit...')

        # Laplace fit is performed on the device the model is located on -> use GPU if available
        if torch.cuda.is_available():
            model.base_model.cuda()

        la: KronLLLaplace = Laplace(model.base_model, 'regression', 'last_layer', 'kron')
        la.fit(datamodule.train_dataloader())
        model.la_model = la

        logger.debug('LA fit done; dumping model to file...')
        # base model can be restored from checkpoint file -> don't serialize it twice
        del model.base_model
        model.base_model = None
        laplace_model_fpath = os.path.join(log_dir, 'laplace_model.gz')
        with gzip.open(laplace_model_fpath, 'wb') as file:
            dill.dump(model, file)

        return TrainResult(interrupted=False)
