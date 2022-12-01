import logging
from typing import Dict, Optional

from pytorch_lightning import LightningDataModule
from torch import nn

import clavicle_ct.litmodel as clavicle_litmodels
from .net.agenet import agenet18_3d, update_model_with_autoencoder_weights
from uncertainty_fae.model import TrainLoadMixin
from uncertainty_fae.util import ModelProvider, TrainConfig

from .data import ClavicleDataModule, get_transforms

logger = logging.getLogger(__name__)

CLAVICLE_LITMODEL_MAPPING: Dict[str, TrainLoadMixin] = {
    "base": clavicle_litmodels.LitClavicle,
    "mc_dropout": clavicle_litmodels.LitClavicleMCDropout,
    "deep_ensemble": clavicle_litmodels.LitClavicleDeepEnsemble,
    "laplace_approx": clavicle_litmodels.LitClavicleLaplace,
    "swag": clavicle_litmodels.LitClavicleSWAG,
}

CLAVICLE_VARIANCE_LITMODEL_MAPPING: Dict[str, TrainLoadMixin] = {
    "base": clavicle_litmodels.LitClavicleVarianceNet,
    "mc_dropout": clavicle_litmodels.LitClavicleVarianceNetMCDropout,
    "deep_ensemble": clavicle_litmodels.LitClavicleVarianceNetDeepEnsemble,
    "laplace_approx": None,
    "swag": None,
}


class ClavicleModelProvider(ModelProvider):
    def __init__(
        self,
        uncertainty_method: str,
        variance_net: bool,
        rescale_boneage: bool = True,
        train_config: Optional[TrainConfig] = None,
        eval_mode: bool = False,
    ) -> None:
        super().__init__()
        self.train_config = train_config
        self.eval_mode = eval_mode
        self.uncertainty_method = uncertainty_method
        self.variance_net = variance_net
        self.rescale_boneage = rescale_boneage

    def get_model(
        self,
        checkpoint=None,
        litmodel_kwargs=dict(),
        checkpoint_kwargs=dict(),
    ) -> nn.Module:
        # Create Network
        output_neurons = 2 if self.variance_net else 1
        net = agenet18_3d(use_dropout=True, use_sex=True, num_classes=output_neurons)

        # Get Litmodel Class
        litmodel_mapping = (
            CLAVICLE_VARIANCE_LITMODEL_MAPPING if self.variance_net else CLAVICLE_LITMODEL_MAPPING
        )
        litmodel_cls = litmodel_mapping[self.uncertainty_method]

        # Create Litmodel
        if checkpoint is not None:
            litmodel = litmodel_cls.load_model_from_disk(
                checkpoint,
                net=net,
                train_config=self.train_config,
                **litmodel_kwargs,
                **checkpoint_kwargs,
            )
        else:
            litmodel = litmodel_cls(
                net=net,
                train_config=self.train_config,
                **litmodel_kwargs,
            )

        # Load pretrained weights, only if no checkpoint is given and not in eval mode
        if not checkpoint and not self.eval_mode:
            defaults = self.train_config.model_configurations_defaults["clavicle_ct"]
            if (
                "autoencoder" in defaults
                and defaults["autoencoder"]
                and "ckpt_path" in defaults["autoencoder"]
            ):
                autoencoder_ckpt_path = defaults["autoencoder"]["ckpt_path"]
                update_model_with_autoencoder_weights(
                    net.state_dict(),
                    autoencoder="agenet18_3d_autoencoder",
                    pretrained_weights=autoencoder_ckpt_path,
                    sex_input=True,
                )
            else:
                logger.warning("No Autoencoder weights given: Training WITHOUT pretrained weights!")

        return litmodel

    def get_lightning_data_module(
        self,
        train_annotation_file: str,
        val_annotation_file: str,
        test_annotation_file: str,
        img_train_base_dir: str = None,
        img_val_base_dir: str = None,
        img_test_base_dir: str = None,
        batch_size: int = 8,
        num_workers: int = 4,
    ) -> LightningDataModule:
        if self.train_config and self.train_config.train_no_augmentation:
            train_data_augmentation_transform = None
            logger.info("Training Data Augmentation DISABLED")
        else:
            train_data_augmentation_transform = get_transforms(augmentation=True)

        datamodule = ClavicleDataModule(
            annotation_file_train=train_annotation_file,
            annotation_file_val=val_annotation_file,
            annotation_file_test=test_annotation_file,
            img_train_base_dir=img_train_base_dir,
            img_val_base_dir=img_val_base_dir,
            img_test_base_dir=img_test_base_dir,
            batch_size=batch_size,
            transforms_train=train_data_augmentation_transform,
            rescale_age=(self.rescale_boneage and not self.eval_mode),
            rebalance_classes=True,
            with_sex_input=True,
            num_workers=num_workers,
            shuffle_train=(not self.eval_mode),
        )

        return datamodule

    @classmethod
    def get_provider(
        cls, train_config: Optional[TrainConfig] = None, eval_mode: bool = False, **kwargs
    ) -> "ClavicleModelProvider":
        return ClavicleModelProvider(train_config=train_config, eval_mode=eval_mode, **kwargs)
