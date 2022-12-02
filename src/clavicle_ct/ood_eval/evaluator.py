import logging
from typing import Callable

import pandas as pd

from uncertainty_fae.evaluation import OutOfDomainEvaluator
from uncertainty_fae.evaluation.util import EvalRunData
from uncertainty_fae.model import UncertaintyAwareModel
from uncertainty_fae.util import EvalRunConfig, ModelProvider


class ClavicleCtOutOfDomainEvaluator(OutOfDomainEvaluator):
    def __init__(
        self,
        data_base_dir: str,
        plot_base_dir: str,
        eval_run_cfg: EvalRunConfig, 
        age_transform: Callable[[pd.Series], pd.Series],
    ) -> None:
        super().__init__(data_base_dir, plot_base_dir, eval_run_cfg, age_transform)

        # self.ood_datasets: dict = self.eval_run_cfg.ood_datasets['clavicle_ct']
        # self.plot_dir = os.path.join(self.plot_base_dir, self.eval_run_cfg.start_time)

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

    def ood_preds_avail(self, eval_cfg_name: str) -> bool:
        return True

    def generate_predictions(
        self,
        eval_cfg_name: str,
        model: UncertaintyAwareModel,
        model_provider: ModelProvider,
    ) -> None:
        pass

    def generate_plots(self, eval_runs_data: dict[str, EvalRunData]) -> None:
        pass
