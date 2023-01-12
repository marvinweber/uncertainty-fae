import csv
import logging
import os
from typing import Callable

import numpy as np
import pandas as pd

from rsna_boneage.data import RSNABoneageDataModule
from uncertainty_fae.evaluation import OutOfDomainEvaluator
from uncertainty_fae.evaluation.util import EvalRunData
from uncertainty_fae.model import UncertaintyAwareModel
from uncertainty_fae.util import EvalRunConfig, ModelProvider

logger = logging.getLogger(__name__)


class RSNABoneAgeOutOfDomainEvaluator(OutOfDomainEvaluator):
    def __init__(
        self,
        data_base_dir: str,
        plot_base_dir: str,
        eval_run_cfg: EvalRunConfig,
        age_to_year_transform: Callable[[pd.Series], pd.Series],
    ) -> None:
        super().__init__(
            data_base_dir,
            plot_base_dir,
            eval_run_cfg,
            age_to_year_transform,
            "rsna_boneage",
        )

    @classmethod
    def get_evaluator(
        cls,
        data_base_dir: str,
        plot_base_dir: str,
        eval_run_cfg: EvalRunConfig,
        age_to_year_transform: Callable[[pd.Series], pd.Series],
    ) -> "OutOfDomainEvaluator":
        evaluator = cls(data_base_dir, plot_base_dir, eval_run_cfg, age_to_year_transform)
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

            dm: RSNABoneageDataModule = model_provider.get_lightning_data_module(
                None,
                ood_cfg["annotations"],
                None,
                img_val_base_dir=ood_cfg["base_dir"],
                batch_size=self.eval_run_cfg.batch_size,
                num_workers=self.eval_run_cfg.dataloader_num_workers,
            )
            dm.setup("validate")  # We always use the validation dataset
            dm.dataset_val.annotations = dm.dataset_val.annotations[:10000]  # max ood samples
            dm.dataset_val.annotations["boneage"] = 0  # we don't need the target to make sense
            # use random sex per sample
            dm.dataset_val.annotations["male"] = np.random.randint(0, 2, size=len(dm.dataset_val))

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
        orig_data_label = "RSNA Bone Age"
        self._generate_uq_comparison_plot(eval_runs_data, orig_data_label)
        self._generate_prediction_comparison_plot(eval_runs_data, orig_data_label)
