import logging
import os
from abc import ABC, abstractmethod
from typing import Callable

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from uncertainty_fae.evaluation.plotting import MEAN_POINT_PROPS

from uncertainty_fae.evaluation.util import EvalRunData
from uncertainty_fae.model import UncertaintyAwareModel
from uncertainty_fae.util import EvalRunConfig, ModelProvider

logger = logging.getLogger(__file__)


class OutOfDomainEvaluator(ABC):
    def __init__(
        self,
        data_base_dir: str,
        plot_base_dir: str,
        eval_run_cfg: EvalRunConfig,
        age_to_year_transform: Callable[[pd.Series], pd.Series],
        ood_dataset_cfg_key: str,
    ) -> None:
        self.data_base_dir = data_base_dir
        self.plot_base_dir = plot_base_dir
        self.eval_run_cfg = eval_run_cfg
        self.age_to_year_transform = age_to_year_transform

        self.ood_datasets: dict[str, dict[str, str]] = self.eval_run_cfg.ood_datasets[
            ood_dataset_cfg_key
        ]
        self.plot_dir = os.path.join(self.plot_base_dir, self.eval_run_cfg.start_time)

    @classmethod
    @abstractmethod
    def get_evaluator(
        cls,
        data_base_dir: str,
        plot_base_dir: str,
        eval_run_cfg: EvalRunConfig,
        age_to_year_transform: Callable[[pd.Series], pd.Series],
    ) -> "OutOfDomainEvaluator":
        """
        Create new instance of an `OutofDomainEvaluator`.

        Args:
            data_base_dir: The directory in which to put predictions.
            plot_base_dir: The directory in which to put evaluation plots.
            eval_run_cfg: The `EvalRunConfig` for the evaluation run.
            age_to_year_transform: The function that should be applied to columns that "represent"
                age (e.g., prediction and uncertainty). Ensures comparison with "normal" dataset in
                the same unit (year, month, etc.). It should transform the values to years.
        """
        ...

    def ood_preds_avail(self, eval_cfg_name: str) -> bool:
        """Check wether Predictions for OOD Eval are available, or not."""
        for ood_name in self.ood_datasets.keys():
            pred_file = self._get_pred_filepath(eval_cfg_name, ood_name)
            if not os.path.exists(pred_file):
                return False  # If single pred file missing: return False
        return True

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

    def _get_pred_filepath(self, eval_cfg_name: str, ood_name: str) -> str:
        """Generate Filepath for OoD Predictions (CSV) based on `eval_cfg_name` and `ood_name`."""
        return os.path.join(self.data_base_dir, f"{eval_cfg_name}_{ood_name}_ood_preds.csv")

    def _load_pred_file(self, eval_cfg_name: str, ood_name: str) -> pd.DataFrame | None:
        """Load Prediction DataFrame and apply Age Transforms."""
        ood_pred_file = self._get_pred_filepath(eval_cfg_name, ood_name)
        if not os.path.isfile(ood_pred_file):
            logger.warning(
                "No OoD-Pred file for %s, %s (%s)", eval_cfg_name, ood_name, ood_pred_file
            )
            return None
        ood_preds = pd.read_csv(ood_pred_file)
        for col in ["prediction", "uncertainty"]:
            ood_preds[col] = self.age_to_year_transform(ood_preds[col])
        return ood_preds

    def _get_fig(self):
        return plt.subplots(figsize=(11, 6), dpi=250)

    def _save_fig(self, fig: Figure, name: str) -> None:
        os.makedirs(self.plot_dir, exist_ok=True)
        fig.tight_layout()
        fig.savefig(os.path.join(self.plot_dir, f"{name}.png"))

    def _generate_uq_comparison_plot(
        self,
        eval_runs_data: dict[str, EvalRunData],
        orig_data_label: str,
    ) -> None:
        """
        Generate a plot comparing the Uncertainties of regular data with OoD datasets.

        Comparison is grouped by `eval_cfg_name` from given `eval_runs_data`.

        Args:
            eval_runs_data: The runs on "normal" data that the OoD predictions should be compared
                with. If no OoD data is available for any of the included eval runs (e.g., the
                predictions have not been created for that UQ method, it is skipped)
            orig_data_label: The name of the "regular" data (in the legend and title).
        """
        violin_positions = []
        violin_means = []
        violin_labels = []
        violin_label_positions = []
        violin_datas = []
        violin_colors = []
        violin_hatches = []
        violin_edge_color = []
        legend_elements = [
            Patch(facecolor="white", edgecolor="black", label=orig_data_label),
        ]

        v_pos = 0
        for eval_cfg_name, eval_run_data in eval_runs_data.items():
            v_pos += 1
            uq_name = eval_run_data["display_name"]
            uq_preds = eval_run_data["prediction_log"]
            color = eval_run_data["color"]

            # Base Entry: UQ with "normal" dataset
            violin_positions.append(v_pos)
            violin_labels.append(uq_name)
            violin_label_positions.append(v_pos + (len(self.ood_datasets) / 2))
            violin_datas.append(uq_preds["uncertainty"].tolist())
            violin_means.append(uq_preds["uncertainty"].mean())
            violin_colors.append(color)
            violin_hatches.append(None)  # no hatch for baseline dataset
            violin_edge_color.append("black")

            for ood_name, ood_cfg in self.ood_datasets.items():
                ood_preds = self._load_pred_file(eval_cfg_name, ood_name)
                if ood_preds is None:
                    continue

                v_pos += 1
                violin_positions.append(v_pos)
                violin_datas.append(ood_preds["uncertainty"].tolist())
                violin_means.append(ood_preds["uncertainty"].mean())
                violin_colors.append(color)
                violin_hatches.append(ood_cfg["hatch"])
                violin_edge_color.append("white")

                if len(legend_elements) < len(self.ood_datasets.items()) + 1:
                    patch = Patch(
                        facecolor="white",
                        edgecolor="grey",
                        hatch=ood_cfg["hatch"],
                        label=ood_cfg["name"],
                    )
                    legend_elements.append(patch)
            v_pos += 1

        fig, ax = self._get_fig()
        vplots = ax.violinplot(
            violin_datas,
            violin_positions,
            showmeans=False,
            showmedians=True,
            widths=1,
        )
        v_patch: PolyCollection
        for v_patch, color, hatch in zip(vplots["bodies"], violin_colors, violin_hatches):
            v_patch.set_facecolor(color)
            if hatch:
                v_patch.set_hatch(hatch)
                v_patch.set_edgecolor(color)
        for partname in ("cbars", "cmins", "cmaxes", "cmedians"):
            vplots[partname].set_edgecolor("black")
            vplots[partname].set_linewidth(1)

        # Mean Markers
        for pos, mean in zip(violin_positions, violin_means):
            ax.plot(pos, mean, **MEAN_POINT_PROPS)

        ax.set_xticks(violin_label_positions)
        ax.set_xticklabels(violin_labels)
        ax.grid(False, axis='x')
        ax.set_ylabel("Uncertainty")
        ax.legend(handles=legend_elements, handleheight=3, handlelength=4, loc="upper left")
        fig.suptitle(f"Out-of-Domain Data Uncertainty Comparison - {orig_data_label}")
        self._save_fig(fig, "uq_comparison")

    def _generate_prediction_comparison_plot(
        self,
        eval_runs_data: dict[str, EvalRunData],
        orig_data_label: str,
    ) -> None:
        """Violin-Plot Comparison of the Predictions for the OOD Datasets by the UQ-Models."""
        violin_positions = []
        violin_labels = []
        violin_datas = []
        violin_colors = []
        violin_hatches = []
        violin_edge_color = []
        legend_elements = []

        v_pos = 0
        for eval_cfg_name, eval_run_data in eval_runs_data.items():
            uq_name = eval_run_data["display_name"]
            color = eval_run_data["color"]

            for ood_name, ood_cfg in self.ood_datasets.items():
                ood_preds = self._load_pred_file(eval_cfg_name, ood_name)
                if ood_preds is None:
                    continue

                v_pos += 1
                violin_positions.append(v_pos)
                violin_labels.append(uq_name)
                violin_datas.append(ood_preds["prediction"].tolist())
                violin_colors.append(color)
                violin_hatches.append(ood_cfg["hatch"])
                violin_edge_color.append("white")

                if len(legend_elements) < len(self.ood_datasets.items()):
                    patch = Patch(
                        facecolor="white",
                        edgecolor="grey",
                        hatch=ood_cfg["hatch"],
                        label=ood_cfg["name"],
                    )
                    legend_elements.append(patch)
            v_pos += 1

        fig, ax = self._get_fig()
        vplots = ax.violinplot(violin_datas, violin_positions, showmedians=True, widths=1)
        v_patch: PolyCollection
        for v_patch, color, hatch in zip(vplots["bodies"], violin_colors, violin_hatches):
            v_patch.set_facecolor(color)
            if hatch:
                v_patch.set_hatch(hatch)
                v_patch.set_edgecolor(color)
        for partname in ("cbars", "cmins", "cmaxes", "cmedians"):
            vplots[partname].set_edgecolor("black")
            vplots[partname].set_linewidth(1)
        ax.set_xticks(violin_positions)
        ax.set_xticklabels(violin_labels, rotation=15)
        ax.set_ylabel("Prediction (Age in Years)")
        ax.legend(handles=legend_elements, handleheight=3, handlelength=4, loc="upper right")
        fig.suptitle(f"Out-of-Domain Data Prediction Comparison - {orig_data_label}")
        self._save_fig(fig, "prediction_comparison")
