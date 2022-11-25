from abc import ABC, abstractmethod

from uncertainty_fae.evaluation.util import EvalRunData
from uncertainty_fae.model import UncertaintyAwareModel
from uncertainty_fae.util import EvalRunConfig, ModelProvider


class OutOfDomainEvaluator(ABC):

    def __init__(self, data_base_dir: str, plot_base_dir: str, eval_run_cfg: EvalRunConfig) -> None:
        self.data_base_dir = data_base_dir
        self.plot_base_dir = plot_base_dir
        self.eval_run_cfg = eval_run_cfg

    @classmethod
    @abstractmethod
    def get_evaluator(
        cls,
        data_base_dir: str,
        plot_base_dir: str,
        eval_run_cfg: EvalRunConfig
    ) -> 'OutOfDomainEvaluator':
        """
        Create new instance of an `OutofDomainEvaluator`.

        Args:
            data_base_dir: The directory in which to put predictions.
            plot_base_dir: The directory in which to put evaluation plots.
            eval_run_cfg: The `EvalRunConfig` for the evaluation run.
        """
        ...

    @abstractmethod
    def ood_preds_avail(self, eval_cfg_name: str) -> bool:
        """Check wether Predictions for OOD Eval are available, or not."""
        ...

    @abstractmethod
    def generate_predictions(
        self,
        eval_cfg_name: str,
        model: UncertaintyAwareModel,
        model_provider: ModelProvider,
    ) -> None:
        """Generate (and store) Out-of-Domain predictions for given model."""
        ...

    @abstractmethod
    def generate_plots(self, eval_runs_data: dict[str, EvalRunData]) -> None:
        """Generate Out-of-Domain comparison plots for all eval runs from given `eval_runs_data`."""
        ...
