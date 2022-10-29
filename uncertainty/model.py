import logging
import os
from typing import Any, Optional, Tuple

import torch
from pytorch_lightning import Callback, LightningDataModule, Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
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

    def forward_without_uncertainty(self, input) -> torch.Tensor:
        # TODO: default implementation: use forward_with_uncertainty and throw away uncertainty
        raise NotImplementedError()

    def evaluate_dataset(self, dataloader: DataLoader):
        raise NotImplementedError()


class TrainLoadMixin:

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @classmethod
    def load_model_from_disk(cls, **kwargs):
        raise NotImplementedError('Method not implemented!')

    @classmethod
    def train_model(cls, log_dir: str, datamodule: LightningDataModule, model: 'TrainLoadMixin',
                    train_config: TrainConfig, is_resume: bool = False,
                    callbacks: Optional[list[Callback]] = None) -> TrainResult:
        """Classmethod to train a model (of cls type, i.e. to train an instance of "itself").

        Args:
            log_dir: The base log_dir where metrics and checkpoints may be saved.
            datamodule: Lightning DataModule containing train, val, (and test) datasets.
            model: The actual instance of `cls` that should be trained.
            train_config: Training configuration.
            is_resume: Whether this training is a resume of a previously started run.
            callbacks: List of additional callbacks to use with a Lightning Trainer.

        Returns:
            The result of the training as `TrainResult`.
        """
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
        lr_monitor_callback = LearningRateMonitor(logging_interval='epoch')
        train_callbacks = [checkpoint_callback, early_stopping_callback, lr_monitor_callback,
                           *(callbacks if callbacks and len(callbacks) > 0 else [])]

        trainer = Trainer(accelerator='gpu', max_epochs=train_config.max_epochs,
                          log_every_n_steps=50, logger=loggers, callbacks=train_callbacks)

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
            interrupted=trainer.interrupted, best_model_path=checkpoint_callback.best_model_path,
            trainer=trainer)
