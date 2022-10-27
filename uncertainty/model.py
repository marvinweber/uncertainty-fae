import logging
import os
from typing import Any, Tuple

import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from util.training import TrainConfig, TrainResult

logger = logging.getLogger(__name__)


class UncertaintyAwareModel:
    """
    A simple wrapper/ interface class that defines a model that is capable of providing uncertainty
    along with its predictions.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward_with_uncertainty(self, input) -> Tuple[torch.Tensor, Any]:
        raise NotImplementedError()

    def evaluate_dataset(self, dataloader: DataLoader):
        raise NotImplementedError()


class TrainMixin:

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @classmethod
    def train_model(cls, log_dir: str, datamodule: LightningDataModule, model: LightningModule,
                    train_config: TrainConfig, is_resume: bool = False) -> TrainResult:
        checkpoint_dir = os.path.join(log_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Organize loggers into directories and "start-time-folders" (to prevent overwriting if
        # a training is resumed).
        tb_logger = pl_loggers.TensorBoardLogger(
            save_dir=log_dir, name='tensorboard', version=train_config.start_time)
        csv_logger = pl_loggers.CSVLogger(
            save_dir=log_dir, name='metrics', version=train_config.start_time)
        loggers = [tb_logger, csv_logger]

        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir, monitor='val_loss', save_last=True,
            save_top_k=train_config.save_top_k_checkpoints, filename='{epoch}-{val_loss:2f}')
        early_stopping_callback = EarlyStopping(
            monitor='val_loss', mode='min', patience=train_config.early_stopping_patience)
        callbacks = [checkpoint_callback, early_stopping_callback]

        trainer = Trainer(accelerator='gpu', max_epochs=train_config.max_epochs,
                          log_every_n_steps=50, logger=loggers, callbacks=callbacks)

        if is_resume:
            # Try to find latest/last checkpoint file and resume the training
            logger.info('RESUMING TRAINING....')
            checkpoint_file = os.path.join(checkpoint_dir, 'last.ckpt')
            if not os.path.exists(checkpoint_file):
                raise RuntimeError('Could not find checkpoint file to resume training '
                                   f'(Path: "{checkpoint_file}"); Aborting!')
            trainer.fit(model, datamodule=datamodule, ckpt_path=checkpoint_file)
        else:
            # Start fresh training
            logger.info('STARTING TRAINING....')
            trainer.fit(model, datamodule=datamodule)

        return TrainResult(
            interrupted=trainer.interrupted, best_model_path=checkpoint_callback.best_model_path)
