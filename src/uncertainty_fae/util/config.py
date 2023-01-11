import argparse
import os
from datetime import datetime
from typing import Any, Optional, Type

import pandas as pd
import yaml
from pytorch_lightning import LightningDataModule, Trainer
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from uncertainty_fae.util.model_provider import ModelProvider

SCRIPT_DESCRIPTIONS = {
    "training": "Generic Model Training Script",
    "evaluation": "Generic Model UQ Evaluatin Script",
    "baseline_model_predictions": "Script to Create (no UQ) Baseline Model Predictions",
}

TRAIN_RESULT_FILENAME = "train_result.yml"


def parse_cli_args(script_type: str) -> dict:
    description = SCRIPT_DESCRIPTIONS[script_type] if script_type in SCRIPT_DESCRIPTIONS else ""
    parser = argparse.ArgumentParser(description=description)

    # General Arguments
    parser.add_argument(
        "--configuration",
        type=str,
        required=False,
        default="./config/models.ma.yml",
        help="Path to the configuration file for available models.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        required=False,
        default=False,
        help="Flag to enable output of DEBUG logs.",
    )
    parser.add_argument("--batch-size", type=int, required=False, default=8)
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        required=False,
        default=4,
        help="Amount of workers to use for dataloading.",
    )

    # Training Arguments
    if script_type == "training":
        parser.add_argument(
            "model_name",
            metavar="MODEL_NAME",
            type=str,
            help="Name of the model to train or evaluate (from model config file).",
        )
        parser.add_argument(
            "--max-epochs",
            type=int,
            default=100,
            required=False,
            help="Maximum Epochs to train.",
        )
        parser.add_argument(
            "--early-stopping-patience",
            type=int,
            default=10,
            required=False,
            help="Patience for EarlyStopping Callback.",
        )
        parser.add_argument(
            "--train-no-augmentation",
            action="store_true",
            required=False,
            default=False,
            help="Disable training data augmentation.",
        )
        parser.add_argument(
            "--save-dir",
            type=str,
            default="/ml_logs",
            required=False,
            help="Directory to save training logs (checkpoints, metrics) to.",
        )
        parser.add_argument(
            "--save-top-k-checkpoints",
            type=int,
            default=2,
            required=False,
            help="Amount of k best checkpoints to save.",
        )
        parser.add_argument(
            "--no-resume",
            action="store_true",
            required=False,
            default=False,
            help="Flag to disable resume of existing training.",
        )
        parser.add_argument(
            "--version",
            type=str,
            required=False,
            default=None,
            help=(
                "Version name of the training (will be timestamp, if not defined). "
                "If the version already exists and --no-resume is NOT set, training "
                "will be resumed if possible."
            ),
        )
        parser.add_argument(
            "--sub-version",
            type=str,
            required=False,
            default=None,
            help=(
                "Subversion can be used to create multiple model versions and is "
                "useful to train a deep ensemble, for example. Note that this is "
                "setting is only allowed if --version is set, too (because "
                "subversions would most likely not be located in the same parent "
                "directory)."
            ),
        )

        # LR Scheduler
        parser.add_argument(
            "--lr-scheduler",
            type=str,
            required=False,
            default=None,
            help="Type of the LR Scheduler to use.",
        )
        parser.add_argument(
            "--lrs-reduce-on-plateau-factor", type=float, required=False, default=0.1
        )
        parser.add_argument(
            "--lrs-reduce-on-plateau-patience", type=int, required=False, default=10
        )

    # Arguments for Evaluation and Creation of Baseline Predictions
    if script_type in ["evaluation", "baseline_model_predictions"]:
        parser.add_argument(
            "--dataset-type",
            choices=["val", "test", "train"],
            default="val",
            required=False,
            help="Which dataset to use (of the Lightning DataModule); Default=val.",
        )
        parser.add_argument(
            "--eval-dir",
            type=str,
            required=False,
            default="/ml_eval",
            help="Base directory to store evaluation results in.",
        )

    # Evaluation Args
    if script_type == "evaluation":
        parser.add_argument(
            "eval_version_name",
            type=str,
            help="Name of the evaluation run (key of the eval config yaml file).",
        )
        parser.add_argument(
            "--eval-configuration",
            type=str,
            required=False,
            default="./config/eval-config.yml",
            help="Path to the file containing the evaluation configuration.",
        )
        parser.add_argument(
            "--eval-only",
            type=str,
            required=False,
            default=None,
            help=(
                "Optionally, comma-separated list of eval config keys/names ("
                "corresponding to the eval-configuration) that should only be "
                "considered in this run (for parallel prediction generation)."
            ),
        )
        parser.add_argument(
            "--only-combined-plots",
            action="store_true",
            default=False,
            required=False,
            help="Whether to only create the combined plots.",
        )
        parser.add_argument(
            "--only-predictions",
            action="store_true",
            default=False,
            required=False,
            help="Whether only predictions should be generated (no plotting).",
        )
        parser.add_argument(
            "--only-plotting",
            action="store_true",
            default=False,
            required=False,
            help=(
                "Whether to only create plots (and never new predictions). This "
                "will make the evaluation skip configs for which predictions are "
                "not available."
            ),
        )
        parser.add_argument(
            "--model-logs-dir",
            required=False,
            default=None,
            help=(
                "Parent directory of all model logs used in the evaluation. "
                "If provided, it will be walked to create best epoch symlinks."
            ),
        )

    # Baseline Prediction Creation Args
    if script_type == "baseline_model_predictions":
        parser.add_argument(
            "model_name",
            metavar="MODEL_NAME",
            type=str,
            help="Name of the model to use (from model config file).",
        )
        parser.add_argument(
            "checkpoint", metavar="CHECKPOINT", type=str, help="Path to the checkpoint file to use."
        )

    args = parser.parse_args()
    return {key: val for key, val in args._get_kwargs()}


class BaseConfig:
    def __init__(self, config_dict: dict) -> None:
        self.config_dict = config_dict
        self.start_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")

        self.batch_size = config_dict["batch_size"]
        self.dataloader_num_workers = config_dict["dataloader_num_workers"]
        self.model_configuration_file = config_dict["configuration"]
        self.model_configurations = None
        self.model_configurations_defaults = None

        self._load_model_configs()
        assert self.model_configurations is not None
        assert self.model_configurations_defaults is not None

    def _load_model_configs(self) -> None:
        with open(self.model_configuration_file, "r") as f:
            configuration = yaml.safe_load(f)

        self.model_configurations = configuration["models"]
        self.model_configurations_defaults = configuration["defaults"]

    def get_annotations_and_base_dir(self, model_name: str, dataset: str) -> tuple[str, str]:
        """
        Get paths to the annotation file and the base directory for given model name.

        Args:
            model_name: The name of the model (from the configuration).
            dataset: The dataset type; e.g., train, val, or test

        Returns:
            A tuple with the annotation file path first and the base dir path second.
        """
        model_config: dict = self.model_configurations[model_name]
        config_defaults = self.model_configurations_defaults
        annotation_file = (
            model_config.get("datasets", {}).get("annotations", {}).get(dataset, {})
            or config_defaults[model_config["data"]]["datasets"]["annotations"][dataset]
        )
        img_base_dir = (
            model_config.get("datasets", {}).get("img_base_dirs", {}).get(dataset, {})
            or config_defaults[model_config["data"]]["datasets"]["img_base_dirs"][dataset]
        )
        return annotation_file, img_base_dir

    def get_model_data_type(self, model_name: str) -> str:
        model_config = self.model_configurations[model_name]
        return model_config["data"]

    def get_model_and_datamodule(
        self,
        model_provider_cls_mapping: dict[str, Type[ModelProvider]],
        model_name: str,
        model_checkpoint: Optional[str] = None,
        eval_cfg_name: Optional[str] = None,
    ) -> tuple[Any, LightningDataModule, ModelProvider]:
        """
        Get Model and Datamodule base on Config and provided Args.

        Args:
            model_provider_cls_mapping: A dict mapping the name of the data type to the respective
                model provider class name. This class will be used to instantiate the model
                provider.
            model_name: The name of the model (according to the config file).
            model_checkpoint: A path to a checkpoint to use to load the model. If not defined, a
                "fresh" model is created.
            eval_cfg_name: Optionally, the name of the evaluation config. Is used to fetch any
                additional model kwargs via `_get_additional_model_providing_kwargs`.

        Returns:
            A tuple (model, datamodule, model_provider) with the created or loaded model first, the
            Lightning data module second, and the model provider third.
        """
        provider_kwargs_adt, litmodel_kwargs_adt = self._get_additional_model_providing_kwargs(
            model_name, eval_cfg_name
        )
        model_config = self.model_configurations[model_name]
        if model_config["data"] not in model_provider_cls_mapping.keys():
            raise ValueError(
                f'Unkown or unsupported Dataset: "{model_config["data"]}" '
                "No Provider Mapping found!"
            )

        # Create Model Provider
        model_provider_cls = model_provider_cls_mapping[model_config["data"]]
        model_provider = model_provider_cls.get_provider(
            **model_config["provider_config"],
            **provider_kwargs_adt,
        )

        # Create Litmodel
        litmodel_kwargs = (
            model_config["litmodel_config"] if "litmodel_config" in model_config else {}
        )
        litmodel_kwargs.update(litmodel_kwargs_adt)
        model = model_provider.get_model(model_checkpoint, litmodel_kwargs=litmodel_kwargs)

        # Annotation files and img base dirs
        train_af, train_d = self.get_annotations_and_base_dir(model_name, "train")
        val_af, val_d = self.get_annotations_and_base_dir(model_name, "val")
        test_af, test_d = self.get_annotations_and_base_dir(model_name, "test")

        # Create Lightning DataModule
        datamodule = model_provider.get_lightning_data_module(
            train_annotation_file=train_af,
            val_annotation_file=val_af,
            test_annotation_file=test_af,
            img_train_base_dir=train_d,
            img_val_base_dir=val_d,
            img_test_base_dir=test_d,
            batch_size=self.batch_size,
            num_workers=self.dataloader_num_workers,
        )
        return model, datamodule, model_provider

    def _get_additional_model_providing_kwargs(
        self, model_name: str, eval_cfg_name: Optional[str] = None
    ) -> tuple[dict, dict]:
        """
        Return tuple of additional kwargs for ModelProvider creation first, and Litmodel second.
        """
        return {}, {}

    def _get_baseline_model_errors_csv_path(
        self, base_dir: str, data_type: str, dataset_type: str
    ) -> str:
        return os.path.join(
            base_dir,
            f"baseline_model_errors_{data_type}_{dataset_type}.csv",
        )


class TrainConfig(BaseConfig):
    def __init__(self, config_dict: dict) -> None:
        super().__init__(config_dict)

        self.max_epochs = config_dict["max_epochs"]
        self.early_stopping_patience = config_dict["early_stopping_patience"]
        self.save_top_k_checkpoints = config_dict["save_top_k_checkpoints"]
        self.train_no_augmentation = config_dict["train_no_augmentation"]
        self.save_dir = config_dict["save_dir"]
        self.no_resume = config_dict["no_resume"]
        self.version = config_dict["version"]
        self.sub_version = config_dict["sub_version"]

        if self.sub_version is not None and self.version is None:
            raise ValueError("You may not define --sub-version without --version!")

    def _get_additional_model_providing_kwargs(
        self, model_name: str, eval_cfg_name: Optional[str] = None
    ) -> tuple[dict, dict]:
        return {"train_config": self}, {}

    def get_lr_scheduler(self, optimizer: Optimizer) -> tuple[Optional[Any], Optional[str]]:
        """Get the LR Scheduler based on given user arguments.

        Args:
            optimizer: The optimizer to wrap with the lr scheduler.

        Returns:
            A tuple with the scheduler first and the metric to monitor (e.g. required by the
            `ReduceLROnPlateau`) second.
        """
        requested_type = self.config_dict["lr_scheduler"]

        if requested_type == "reduce_lr_on_plateau":
            factor = self.config_dict["lrs_reduce_on_plateau_factor"]
            patience = self.config_dict["lrs_reduce_on_plateau_patience"]
            scheduler = ReduceLROnPlateau(
                optimizer, factor=factor, patience=patience, threshold=1e-4
            )
            return scheduler, "val_loss"

        return None, None


class EvalRunConfig(BaseConfig):
    def __init__(self, config_dict: dict) -> None:
        super().__init__(config_dict)

        self.eval_version_name = config_dict["eval_version_name"]
        self.dataset_type = config_dict["dataset_type"]
        self.eval_configuration_file = config_dict["eval_configuration"]
        self.eval_configuration: dict
        self.ood_datasets: dict
        self.eval_dir = config_dict["eval_dir"]
        self.only_combined_plots: bool = config_dict["only_combined_plots"]
        self.only_predictions: bool = config_dict["only_predictions"]
        self.only_plotting: bool = config_dict["only_plotting"]
        self.model_logs_dir: Optional[str] = config_dict["model_logs_dir"]

        self._load_eval_configuration()
        assert isinstance(self.eval_configuration, dict)

        if isinstance(config_dict["eval_only"], str) and len(config_dict["eval_only"]) > 0:
            self.eval_only = [
                cfg_name
                for cfg_name in config_dict["eval_only"].split(",")
                if cfg_name in self.eval_configuration.keys()
            ]
        else:
            self.eval_only: list[str] = list(self.eval_configuration.keys())

        assert not (
            self.only_predictions and self.only_plotting
        ), "Cannot --only-predictions and --only-plotting at the same time!"

    def _load_eval_configuration(self) -> None:
        with open(self.eval_configuration_file, "r") as f:
            full_eval_configuration = yaml.safe_load(f)
        self.eval_configuration = full_eval_configuration["evaluations"][self.eval_version_name]
        self.ood_datasets = full_eval_configuration["ood_datasets"]

    def _get_additional_model_providing_kwargs(
        self, model_name: str, eval_cfg_name: Optional[str] = None
    ) -> tuple[dict, dict]:
        if eval_cfg_name not in self.eval_configuration:
            raise ValueError("EvalRunConfig requires eval_cfg_name!")
        cfg = self.eval_configuration[eval_cfg_name]
        litmodel_kwargs = cfg["litmodel_config"] if "litmodel_config" in cfg else {}
        return {"eval_mode": True}, litmodel_kwargs

    def get_eval_dataloader(self, datamodule: LightningDataModule):
        if self.dataset_type == "train":
            return datamodule.train_dataloader()
        elif self.dataset_type == "val":
            return datamodule.val_dataloader()
        elif self.dataset_type == "test":
            return datamodule.test_dataloader()
        else:
            raise ValueError('EvalRunConfig: Invalid DataSet Type: "%s"!', self.dataset_type)

    def get_baseline_error_dataframe(self, data_type: str) -> pd.DataFrame | None:
        """
        Load the CSV file containing errors of baseline (no UQ) model into Pandas DataFrame.

        Args:
            data_type: Data type of data to get the errors for.

        Returns:
            A `DataFrame` of the errors of the baseline model or, if the corresponding CSV file is
            not found, `None`.
        """
        filepath = self._get_baseline_model_errors_csv_path(
            self.eval_dir, data_type, self.dataset_type
        )
        if not os.path.isfile(filepath):
            return None
        df = pd.read_csv(filepath)
        return df

    def get_mean_predictor_error_dataframe(self, data_type: str) -> pd.DataFrame | None:
        filepath = os.path.join(
            self.eval_dir,
            f"mean_predictor_model_errors_{data_type}_{self.dataset_type}.csv",
        )
        if not os.path.isfile(filepath):
            return None
        df = pd.read_csv(filepath)
        return df


class BaselinePredictionConfig(BaseConfig):
    def __init__(self, config_dict: dict) -> None:
        super().__init__(config_dict)
        self.dataset_type: str = self.config_dict["dataset_type"]
        self.model_name: str = self.config_dict["model_name"]
        self.eval_dir: str = self.config_dict["eval_dir"]
        self.checkpoint: str = self.config_dict["checkpoint"]

    def _get_additional_model_providing_kwargs(
        self, model_name: str, eval_cfg_name: Optional[str] = None
    ) -> tuple[dict, dict]:
        return {"eval_mode": True}, {}

    def get_baseline_model_errors_csv_path(self) -> str:
        return self._get_baseline_model_errors_csv_path(
            self.eval_dir,
            self.get_model_data_type(self.model_name),
            self.dataset_type,
        )


class TrainResult:
    def __init__(
        self,
        interrupted: bool,
        best_model_path: str = None,
        additional_info: dict = None,
        trainer: Optional[Trainer] = None,
    ) -> None:
        self.interrupted = interrupted
        self.best_model_path = best_model_path
        self.additional_info = additional_info
        self.trainer = trainer
