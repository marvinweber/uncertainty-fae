import csv
import logging
import os
from typing import Callable

import pandas as pd

from clavicle_ct.data import ClavicleDataModule
from uncertainty_fae.evaluation import OutOfDomainEvaluator
from uncertainty_fae.evaluation.util import EvalRunData
from uncertainty_fae.model import UncertaintyAwareModel
from uncertainty_fae.util import EvalRunConfig, ModelProvider

logger = logging.getLogger(__file__)


class ClavicleCtOutOfDomainEvaluator(OutOfDomainEvaluator):
    def __init__(
        self,
        data_base_dir: str,
        plot_base_dir: str,
        eval_run_cfg: EvalRunConfig,
        age_transform: Callable[[pd.Series], pd.Series],
    ) -> None:
        super().__init__(data_base_dir, plot_base_dir, eval_run_cfg, age_transform, "clavicle_ct")

    @classmethod
    def get_evaluator(
        cls,
        data_base_dir: str,
        plot_base_dir: str,
        eval_run_cfg: EvalRunConfig,
        age_transform: Callable[[pd.Series], pd.Series],
    ) -> "ClavicleCtOutOfDomainEvaluator":
        evaluator = cls(data_base_dir, plot_base_dir, eval_run_cfg, age_transform)
        return evaluator

    def generate_predictions(
        self,
        eval_cfg_name: str,
        model: UncertaintyAwareModel,
        model_provider: ModelProvider,
    ) -> None:
        for ood_name, ood_cfg in self.ood_datasets.items():
            logger.info("Next OOD Dataset: %s", ood_name)
            pred_file = self._get_pred_filepath(eval_cfg_name, ood_name)

            if os.path.exists(pred_file):
                logger.info("Skipping, as prediction file already available...")
                continue

            dm: ClavicleDataModule = model_provider.get_lightning_data_module(
                None,
                ood_cfg["annotations"],
                None,
                img_val_base_dir=ood_cfg["base_dir"],
                batch_size=self.eval_run_cfg.batch_size,
                num_workers=self.eval_run_cfg.dataloader_num_workers,
            )
            dm.setup("validate")  # We always use the validation dataset

            results = model.evaluate_dataset(dm.val_dataloader())
            score, predictions, targets, errors, uncertainties, metrics = results

            # Prediction and Uncertainty Stats
            os.makedirs(self.data_base_dir, exist_ok=True)
            with open(pred_file, "w") as file:
                writer = csv.writer(file)
                writer.writerow(["index", "prediction", "uncertainty"])
                writer.writerows(
                    zip(range(len(predictions)), predictions.tolist(), uncertainties.tolist())
                )

    def generate_plots(self, eval_runs_data: dict[str, EvalRunData]) -> None:
        orig_data_label = "Clavicle CT"
        self._generate_uq_comparison_plot(eval_runs_data, orig_data_label, "lower right")
        self._generate_prediction_comparison_plot(eval_runs_data, orig_data_label)
