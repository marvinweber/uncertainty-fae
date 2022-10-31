import logging
from typing import Any, Optional, Union, cast

import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.utilities.types import LRSchedulerConfig, _LRScheduler
from torch.optim.swa_utils import SWALR

from swa_gaussian.posteriors.swag import SWAG

logger = logging.getLogger(__name__)


class SWAGaussianCallback(Callback):

    def __init__(
            self,
            swag_model: SWAG,
            swa_lrs: Union[float, list[float]],
            swa_start_epoch: int = 31,
            annealing_epochs: int = 10,
            annealing_strategy: str = 'linear',
    ) -> None:
        super().__init__()
        self.swag_model = swag_model
        self._initialized = False
        self._latest_update_epoch = -1

        self._swa_start_epoch = swa_start_epoch
        self._annealing_epochs = annealing_epochs
        self._annealing_strategy = annealing_strategy
        self._swa_lrs = swa_lrs

        self._swa_scheduler: Optional[_LRScheduler] = None
        self._scheduler_state: Optional[dict] = None

    @property
    def swa_start(self) -> int:
        """SWA Start Epoch zero based."""
        assert isinstance(self._swa_start_epoch, int)
        return max(self._swa_start_epoch - 1, 0)  # 0-based

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        super().on_fit_start(trainer, pl_module)

        if trainer.max_epochs > self._swa_start_epoch:  # max_epochs is not zero based
            logger.warning('SWA will not be utilized as swa_start_epoch > max_epochs!')

        assert len(trainer.optimizers) == 1, \
            'SWA(G) currently works with only 1 `optimizer`.'

        assert len(trainer.lr_scheduler_configs) <= 1, \
            'SWA(G) currently not supported for more than 1 `lr_scheduler`.'

        if self._scheduler_state is not None:
            self._clear_schedulers(trainer)

        self.swag_model.to(pl_module.device)
        logger.debug('SWAG model on target device; Callback initialized for training.')

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch = trainer.current_epoch
        should_perform_swa_update = epoch >= self.swa_start

        # Initalization
        if not self._initialized and should_perform_swa_update:
            self._initialized = True

            optimizer = trainer.optimizers[0]
            if isinstance(self._swa_lrs, float):
                self._swa_lrs = [self._swa_lrs] * len(optimizer.param_groups)

            for lr, group in zip(self._swa_lrs, optimizer.param_groups):
                group['initial_lr'] = lr

            self._swa_scheduler = cast(
                _LRScheduler,
                SWALR(
                    optimizer,
                    swa_lr=self._swa_lrs,  # type: ignore[arg-type]
                    anneal_epochs=self._annealing_epochs,
                    anneal_strategy=self._annealing_strategy,
                    last_epoch=trainer.max_epochs if self._annealing_strategy == 'cos' else -1,
                ),
            )
            if self._scheduler_state is not None:
                # Restore scheduler state from checkpoint
                self._swa_scheduler.load_state_dict(self._scheduler_state)
            elif trainer.current_epoch > self.swa_start:
                # Log a warning if we're initializing after start without any checkpoint data,
                # as behaviour will be different compared to having checkpoint data.
                logger.warning(
                    "SWA is initializing after swa_start without any checkpoint data. "
                    "This may be caused by loading a checkpoint from an older version of PyTorch Lightning."
                )

            # We assert that there is only one optimizer on fit start, so know opt_idx is always 0
            default_scheduler_cfg = LRSchedulerConfig(self._swa_scheduler, opt_idx=0)
            assert default_scheduler_cfg.interval == 'epoch' and default_scheduler_cfg.frequency == 1

            if trainer.lr_scheduler_configs:
                scheduler_cfg = trainer.lr_scheduler_configs[0]
                if scheduler_cfg.interval != 'epoch' or scheduler_cfg.frequency != 1:
                    logger.warning(
                        f'SWA(G) is currently only supported every epoch. Found {scheduler_cfg}')
                logger.warning(
                    f'Swapping scheduler `{scheduler_cfg.scheduler.__class__.__name__}`'
                    f' for `{self._swa_scheduler.__class__.__name__}`'
                )
                trainer.lr_scheduler_configs[0] = default_scheduler_cfg
            else:
                trainer.lr_scheduler_configs.append(default_scheduler_cfg)

        if should_perform_swa_update and trainer.current_epoch > self._latest_update_epoch:
            logger.info('Starting SWA model update (collecting model).')
            self.swag_model.collect_model(pl_module)
            self._latest_update_epoch = trainer.current_epoch
            logger.debug('SWA Calculation done.')

    def state_dict(self) -> dict[str, Any]:
        return {
            'swag_model_state': self.swag_model.state_dict(),
            'latest_update_epoch': self._latest_update_epoch,
            'scheduler_state': (None
                                if self._swa_scheduler is None
                                else self._swa_scheduler.state_dict()),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.swag_model.load_state_dict(state_dict['swag_model_state'])
        self._latest_update_epoch = state_dict['latest_update_epoch']
        self._scheduler_state = state_dict['scheduler_state']

    @staticmethod
    def _clear_schedulers(trainer: pl.Trainer) -> None:
        # If we have scheduler state saved, clear the scheduler configs so that we don't try to
        # load state into the wrong type of schedulers when restoring scheduler checkpoint state.
        # We'll configure the scheduler and re-load its state in on_train_epoch_start.
        # Note that this relies on the callback state being restored before the scheduler state is
        # restored, and doesn't work if restore_checkpoint_after_setup is True, but at the time of
        # writing that is only True for deepspeed which is already not supported by SWA.
        # See https://github.com/PyTorchLightning/pytorch-lightning/issues/11665 for background.
        if trainer.lr_scheduler_configs:
            assert len(trainer.lr_scheduler_configs) == 1
            trainer.lr_scheduler_configs.clear()
