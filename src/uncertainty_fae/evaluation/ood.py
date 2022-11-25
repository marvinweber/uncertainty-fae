from abc import ABC, abstractmethod

from uncertainty_fae.evaluation.util import EvalRunData
from uncertainty_fae.model import UncertaintyAwareModel
from uncertainty_fae.util import EvalRunConfig, ModelProvider


class OutOfDomainEvaluator(ABC):

    def __init__(self, base_dir: str, eval_run_cfg: EvalRunConfig) -> None:
        self.base_dir = base_dir
        self.eval_run_cfg = eval_run_cfg

    @classmethod
    @abstractmethod
    def get_evaluator(cls, base_dir: str, eval_run_cfg: EvalRunConfig) -> 'OutOfDomainEvaluator':
        """
        Create new instance of an `OutofDomainEvaluator`.

        Args:
            base_dir: The directory in which to put predictions and plots.
            eval_run_cfg: The `EvalRunConfig` for the evaluation run.
        """
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
