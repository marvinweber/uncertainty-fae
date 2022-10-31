import logging
import os
from typing import Any, Optional, Tuple

import torch
from pytorch_lightning import Callback, LightningDataModule, Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor, ModelCheckpoint,
                                         TQDMProgressBar)
from torch import Tensor
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

    def forward_with_uncertainty(self, input) -> Tuple[Tensor, Tensor]:
        """Forward a batch/input and return results including uncertainty.

        Returns:
            A tuple with the results ("mean") values first, and the uncertainty values (e.g. std)
            second. The size on each of those two tensors corresponds to the input (batch) size.

        Example:
            For an input of batch size = 2 the result may look like shown below:

            >>> res = foo.forward_with_uncertainty(x)
            >>> res[0]  # mean values (predictions)
            >>> tensor([0.92, 0.51])
            >>> res[1]  # uncertainties
            >>> tensor([0.01, 0.02])
        """
        raise NotImplementedError()

    def forward_without_uncertainty(self, input) -> torch.Tensor:
        # TODO: default implementation: use forward_with_uncertainty and throw away uncertainty
        raise NotImplementedError()

    def evaluate_dataset(
        self,
        dataloader: DataLoader
    ) -> tuple[Any, Tensor, Tensor, Tensor, Tensor, list[dict], dict[str, Any]]:
        """Evaluate the given given dataset (dataloader).

        TODO docs/ explanation

        Returns:
            A tuple with the following entries

            - score: a metric describing the "score" of the model w.r.t. the dataloader (e.g., mean
                abs error)
            - predictions: A tensor containing the predictions for every sample of the loader.
            - targets: A tensor containing the targets (ground truth) for every sample of the
                loader.
            - errors: A tensor containing the error for each single sample of the loader (e.g., the
                mean abs error for every sample).
            - uncertainties: A tensor containing the uncertainty value for each single sample of the
                loader (e.g., the std of many predictions per sample).
            - sample_stats: A list of dictionaries containing (arbitrary) stats for each sample of
                the dataloader.
            - additional_stats: A dictionary containing any (additional) stats.
        """
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

        progress_bar = TQDMProgressBar(refresh_rate=25)
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir, monitor='val_loss', save_last=True,
            save_top_k=train_config.save_top_k_checkpoints, filename='{epoch}-{val_loss:2f}')
        early_stopping_callback = EarlyStopping(
            monitor='val_loss', mode='min', patience=train_config.early_stopping_patience)
        lr_monitor_callback = LearningRateMonitor(logging_interval='epoch')
        train_callbacks = [checkpoint_callback, early_stopping_callback, lr_monitor_callback,
                           progress_bar, *(callbacks if callbacks and len(callbacks) > 0 else [])]

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
