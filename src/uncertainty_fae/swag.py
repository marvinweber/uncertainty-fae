import logging
from typing import Optional
import pytorch_lightning as pl
from pytorch_lightning import Callback
from torch.utils.data import DataLoader

from uncertainty_fae.model import UncertaintyAwareModel

logger = logging.getLogger(__name__)


class SwagEvalCallback(Callback):

    def __init__(
            self,
            swag_model: UncertaintyAwareModel,
            swa_start_epoch: int,
            metrics_to_log: list[str],
            swa_eval_frequency: int = 1,
            validation_dataloader: Optional[DataLoader] = None,
    ) -> None:
        super().__init__()
        self.swag_model = swag_model
        self.swa_start_epoch = swa_start_epoch
        self.metrics_to_log = metrics_to_log
        self.swa_eval_frequency = swa_eval_frequency
        self.validation_dataloader = validation_dataloader
        self._checks_performed = False

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        super().on_validation_start(trainer, pl_module)

        # Ensure we have a validation dataloader
        if not self.validation_dataloader and not self._checks_performed:
            assert trainer.val_dataloaders, \
                ('SwagEvalCallback either needs to be initialized with a `validation_loader`, or '
                 'the trainer must have >= 1 validation loaders')
            if len(trainer.val_dataloaders) > 1:
                logger.warning(
                    'SwagEvalCallback currently only uses the first trainer validation dataloader!')
            self.validation_dataloader = trainer.val_dataloaders[0]
        self._checks_performed = True

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch = trainer.current_epoch
        swa_started = epoch >= (self.swa_start_epoch - 1)
        should_eval = (epoch - (self.swa_start_epoch - 1)) % self.swa_eval_frequency == 0
        if not swa_started or not should_eval:
            return

        logger.info('Running SWAG Evaluation...')
        score, _, _, _, _, metrics = self.swag_model.evaluate_dataset(self.validation_dataloader)
        logs = {f'swag_val_{key}': val
                for key, val in metrics.items() if key in self.metrics_to_log}
        for l in trainer.loggers:
            l.log_metrics(logs, step=trainer.fit_loop.epoch_loop._batches_that_stepped)
        logger.info(f'SWAG Evaluation done; Score={score}')
