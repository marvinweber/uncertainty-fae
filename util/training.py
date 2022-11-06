import argparse
from typing import Any, Optional

from pytorch_lightning import Trainer
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau


def parse_cli_args(type: str) -> argparse.Namespace:
    if type == 'training':
        description = 'Training of RSNA Boneage Models'
    else:
        description = ''

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('model_name', metavar='MODEL_NAME', type=str,
                        help='Name of the model to train or evaluate (from model config file).')
    parser.add_argument('--configuration', metavar='CONFIGURATION', type=str, required=False,
                        default='./config/models.yml',
                        help='Path to the configuration file for available models.')
    parser.add_argument('--max-epochs', metavar='MAX_EPOCHS', type=int, default=100, required=False,
                        help='Maximum Epochs to train.')
    parser.add_argument('--early-stopping-patience', metavar='EARLY_STOPPING_PATIENCE', type=int,
                        default=10, required=False, help='Patience for EarlyStopping Callback.')
    parser.add_argument('--save-dir', metavar='SAVE_DIR', type=str, default='train_logs',
                        required=False,
                        help='Directory to save training logs (checkpoints, metrics) to.')
    parser.add_argument('--save-top-k-checkpoints', metavar='SAVE_TOP_K_CHECKPOINTS', type=int,
                        default=2, required=False, help='Amount of k best checkpoints to save.')
    parser.add_argument('--no-resume', action='store_true', required=False, default=False,
                        help='Flag to disable resume of existing training.')
    parser.add_argument('--version', metavar='VERSION', type=str, required=False, default=None,
                        help='Version name of the training (will be timestamp, if not defined). '
                             'If the version already exists and --no-resume is NOT set, training '
                             'will be resumed if possible.')
    parser.add_argument('--sub-version', type=str, required=False, default=None,
                        help='Subversion can be used to create multiple model versions and is '
                             'useful to train a deep ensemble, for example. Note that this is '
                             'setting is only allowed if --version is set, too (because '
                             'subversions would most likely not be located in the same parent '
                             'directory).')
    parser.add_argument('--debug', action='store_true', required=False, default=False,
                        help='Flag to enable output of DEBUG logs.')
    parser.add_argument('--batch-size', type=int, required=False, default=8)
    parser.add_argument('--dataloader-num-workers', type=int, required=False, default=4,
                        help='Amount of workers to use for dataloading.')
    # LR Scheduler
    parser.add_argument('--lr-scheduler', type=str, required=False, default=None,
                        help='Type of the LR Scheduler to use.')
    parser.add_argument('--lrs-reduce-on-plateau-factor', type=float, required=False, default=0.1)
    parser.add_argument('--lrs-reduce-on-plateau-patience', type=int, required=False, default=10)

    return parser.parse_args()


class TrainConfig():

    def __init__(
        self,
        args: argparse.Namespace,
        max_epochs: int,
        early_stopping_patience: int,
        save_dir: str,
        start_time: str,
        batch_size: int = 8,
        save_top_k_checkpoints: int = 2,
        no_resume: bool = False,
        version: str = None,
        sub_version: str = None,
        dataloader_num_workers: int = 4,
    ) -> None:
        self.args = args

        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.save_dir = save_dir
        self.start_time = start_time
        self.batch_size = batch_size
        self.save_top_k_checkpoints = save_top_k_checkpoints
        self.no_resume = no_resume
        self.version = version
        self.sub_version = sub_version
        self.dataloader_num_workers = dataloader_num_workers

        if self.sub_version is not None and self.version is None:
            raise ValueError('You may not define --sub-version without --version!')

    def get_lr_scheduler(self, optimizer: Optimizer) -> tuple[Any, str]:
        """Get the LR Scheduler based on given user arguments.

        Args:
            optimizer: The optimizer to wrap with the lr scheduler.
        
        Returns:
            A tuple with the scheduler first and the metric to monitor (e.g. required by the
            `ReduceLROnPlateau`) second.
        """
        requested_type = self.args.lr_scheduler

        if requested_type == 'reduce_lr_on_plateau':
            factor = self.args.lrs_reduce_on_plateau_factor
            patience = self.args.lrs_reduce_on_plateau_patience
            scheduler = ReduceLROnPlateau(
                optimizer, factor=factor, patience=patience, threshold=1e-4)
            return scheduler, 'val_loss'


class TrainResult():

    def __init__(self, interrupted: bool, best_model_path: str = None,
                 additional_info: dict = None, trainer: Optional[Trainer] = None) -> None:
        self.interrupted = interrupted
        self.best_model_path = best_model_path
        self.additional_info = additional_info
        self.trainer = trainer
