import logging
import os
from typing import Any, Optional

import torch
import tqdm
from pytorch_lightning import Callback, LightningDataModule, Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor, ModelCheckpoint,
                                         TQDMProgressBar)
from torch import Tensor
from torch.utils.data import DataLoader

from uncertainty_fae.util import TrainConfig, TrainResult

logger = logging.getLogger(__name__)


class ForwardMetrics():
    def __init__(self, preds_distinct: Optional[Tensor] = None) -> None:
        self.preds_distinct = preds_distinct


class EvaluationMetrics():
    def __init__(
        self,
        preds_distinct: Optional[list[Tensor]] = None,
        mean_uncertainty: Optional[Tensor] = None,
        distinct_model_errors: Optional[list[Tensor]] = None,
    ) -> None:
        self.preds_distinct = preds_distinct
        self.mean_uncertainty = mean_uncertainty
        self.distinct_model_errors = distinct_model_errors


class UncertaintyAwareModel:
    """
    A simple wrapper/ interface class that defines a model that is capable of providing uncertainty
    along with its predictions.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.train_dataloader: DataLoader = None
        self.val_dataloader: DataLoader = None

    def set_dataloaders(
            self,
            train_dataloader: DataLoader = None,
            val_dataloader: DataLoader = None,
    ) -> None:
        """Set train and val dataloader on this model.
        
        Some UQ methods require the training dataset (SWAG) to update batch norm layers.
        """
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

    def forward_with_uncertainty(self, input) -> tuple[Tensor, Tensor, ForwardMetrics]:
        """Forward a batch/input and return results including uncertainty.

        Note: Returned values should be located on the CPU device.

        Args:
            input: The input to process.

        Returns:
            A tuple with the results ("mean") values first, and the uncertainty values (e.g. std)
            second. The size on each of those two tensors corresponds to the input (batch) size.
            Third, the method returns `ForwardMetrics`, with (optional) more information. Each entry
            in the `ForwardMetrics` object (if a list or Tensor) should have length == batch_size,
            where each item corresponds to the item from the batch with same index.

        Example:
            For an input of batch size = 2 the result may look like shown below:

            >>> res = foo.forward_with_uncertainty(x)
            >>> res[0]  # mean values (predictions)
            >>> tensor([0.92, 0.51])
            >>> res[1]  # uncertainties
            >>> tensor([0.01, 0.02])
        """
        raise NotImplementedError()

    def forward_without_uncertainty(self, input) -> Tensor:
        return self.forward_with_uncertainty(input)[0]

    def evaluate_dataset(
        self, dataloader: DataLoader
    ) -> tuple[Any, Tensor, Tensor, Tensor, Tensor, EvaluationMetrics]:
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
            - evaluation_metrics: Aadditional stats.
        """
        raise NotImplementedError()


def uam_evaluate_dataset_default(
    model: UncertaintyAwareModel,
    device,
    dataloader: DataLoader,
) -> tuple[Any, Tensor, Tensor, Tensor, Tensor, EvaluationMetrics]:
    """Default implementation for `UncertaintyAwareModel::evaluate_dataset().

    TODO Documentation
    """
    targets = []
    preds_mean = []
    preds_std = []
    preds_var = []
    preds_distinct = []

    data_iterator = tqdm.tqdm(
        dataloader, desc=f'Evaluation Progress', total=len(dataloader), leave=False)
    for input, target in data_iterator:
        targets.append(target)
        mean, std, batch_metrics = model.forward_with_uncertainty(input.to(device))
        preds_mean.append(mean)
        preds_std.append(std)
        if batch_metrics.preds_distinct:
            preds_distinct.append(batch_metrics.preds_distinct)

    targets = torch.cat(targets)
    preds_mean = torch.cat(preds_mean)
    preds_std = torch.cat(preds_std)
    preds_abs_errors = torch.abs((preds_mean - targets))
    mae = torch.mean(preds_abs_errors)

    eval_metrics = EvaluationMetrics(mean_uncertainty=torch.mean(preds_std))

    if len(preds_distinct) > 0:
        preds_distinct = [p for batch in preds_distinct for p in batch]
        assert len(preds_distinct) == len(preds_mean)  # Verify we got distincts for all samples
        eval_metrics.preds_distinct = preds_distinct

    return mae, preds_mean, targets, preds_abs_errors, preds_std, eval_metrics


class TrainLoadMixin:

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @classmethod
    def load_model_from_disk(cls, checkpoint_path, **kwargs):
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

        trainer = Trainer(
            accelerator='gpu',
            max_epochs=train_config.max_epochs,
            log_every_n_steps=50,
            logger=loggers,
            callbacks=train_callbacks,
            reload_dataloaders_every_n_epochs=1,
        )

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
