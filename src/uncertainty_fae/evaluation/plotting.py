import os
from datetime import datetime
from typing import Callable, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from uncertainty_fae.evaluation.calibration import (
    QUANTILE_SIGMA_ENV_SCALES,
    observation_share_per_prediction_interval,
)
from uncertainty_fae.evaluation.util import EvalRunData, apply_df_age_transform, sort_min_swaps

TARGET_COLOR = "green"
"""Color to use in plots for the target (ground truth)."""

BASELINE_MODEL_COLOR = "black"
"""Color to use in plots for the baseline model, i.e., model without UQ."""

MEAN_POINT_PROPS = dict(
    marker="D",
    markeredgecolor="black",
    markerfacecolor="firebrick",
)

MEAN_LEGEND_ENTRY_PROPS = dict(**MEAN_POINT_PROPS, color=(0, 0, 0, 0), label="Mean")

BOXPLOT_FLIER_PROPS = dict(marker="x", markersize=5, markeredgecolor="black")


class EvalPlotGenerator:
    def __init__(
        self,
        eval_runs_data: dict[str, EvalRunData],
        img_save_dir: str,
        img_ext: str = "png",
        plt_style: Optional[str] = None,
        img_prepend_str: str = "",
        img_with_timestamp: bool = False,
        undo_age_to_year_transform: Optional[Callable[[pd.Series], pd.Series]] = None,
        baseline_model_error_df: Optional[pd.DataFrame] = None,
        mean_predictor_model_error_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        TODO: Docs
        """
        self.eval_runs_data = eval_runs_data
        self.img_save_dir = img_save_dir
        self.img_ext = img_ext
        self.img_prepend_str = img_prepend_str
        self.img_with_timestamp = img_with_timestamp
        self.ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.baseline_model_error_df = baseline_model_error_df
        self.mean_predictor_model_error_df = mean_predictor_model_error_df

        self.undo_age_to_year_transform = undo_age_to_year_transform
        if self.undo_age_to_year_transform is None:
            self.undo_age_to_year_transform = lambda x: x

        if not plt_style:
            seaborn.set_theme(context="paper")
        else:
            plt.style.use(plt_style)

    def plot_age_distribution(self, eval_cfg_name: str, bins=25) -> None:
        df = self.eval_runs_data[eval_cfg_name]["prediction_log"]
        p_color = self.eval_runs_data[eval_cfg_name]["color"]
        fig, ax = self._get_figure(
            title="Distribution of Predicted and Ground Truth Age",
            derive_suptitle_from_cfg=eval_cfg_name,
        )
        ax.set_xlabel("(Predicted) Age / (Years)")
        ax.hist(
            [df["prediction"].tolist(), df["target"].tolist()],
            label=["Predicted", "Ground Truth"],
            color=[p_color, TARGET_COLOR],
            bins=bins,
        )
        ax.legend()
        self._save_figure(fig, "age_distribution")

    def plot_uncertainty_by_age(self, eval_cfg_name: str) -> None:
        df = self.eval_runs_data[eval_cfg_name]["prediction_log"]
        target_min = int(np.floor(df["target"].min()))
        target_max = int(np.ceil(df["target"].max()))
        age_bins = np.linspace(start=target_min, stop=target_max, num=target_max - target_min + 1)
        age_bins_series = pd.cut(df["target"], bins=age_bins)
        df_binned_by_age = df.groupby(age_bins_series).agg({"uncertainty": [list, "mean"]}).dropna()

        bin_values = []
        positions = []
        widths = []
        means = []
        for bin, val in df_binned_by_age.iterrows():
            bin_values.append(val["uncertainty"]["list"])
            means.append(val["uncertainty"]["mean"])
            width = bin.right - bin.left
            positions.append((width / 2) + bin.left)
            widths.append(width)

        fig, ax = self._get_figure(
            title="Uncertainty by Age",
            derive_suptitle_from_cfg=eval_cfg_name,
        )

        vplot = ax.violinplot(bin_values, positions=positions, widths=widths, showmedians=False)
        for patch in vplot["bodies"]:
            patch.set_facecolor(self.eval_runs_data[eval_cfg_name]["color"])
        for partname in ("cbars", "cmins", "cmaxes"):
            vplot[partname].set_edgecolor("black")
            vplot[partname].set_linewidth(1)

        for pos, mean in zip(positions, means):
            ax.plot(pos, mean, **MEAN_POINT_PROPS)

        ax.legend(handles=[Line2D([0], [0], **MEAN_LEGEND_ENTRY_PROPS)])
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Age / (Years) [Bin Width = 1 Year]")
        ax.set_ylabel("Uncertainty")
        ax.set_xticks(age_bins)
        self._save_figure(fig, "uncertainty_by_age")

    def plot_uncertainty_by_error(
        self,
        eval_cfg_name: str,
        bin_width: float = 0.5,
        with_aucs: bool = False,
    ) -> None:
        color = self.eval_runs_data[eval_cfg_name]["color"]
        df = self.eval_runs_data[eval_cfg_name]["prediction_log"]

        error_bins = self._get_error_bins([eval_cfg_name], bin_width=bin_width)
        error_bins_series = pd.cut(df["error"], bins=error_bins)
        df_grouped_by_error_bins = (
            df.groupby(error_bins_series).agg({"uncertainty": [list, "mean"]}).dropna()
        )

        bin_values = []
        positions = []
        widths = []
        means = []
        for bin, val in df_grouped_by_error_bins.iterrows():
            bin_values.append(val["uncertainty"]["list"])
            means.append(val["uncertainty"]["mean"])
            width = bin.right - bin.left
            positions.append((width / 2) + bin.left)
            widths.append(width)

        fig, ax = self._get_figure(
            title="Uncertainty by Error",
            derive_suptitle_from_cfg=eval_cfg_name,
        )
        legend_handles = [Line2D([0], [0], **MEAN_LEGEND_ENTRY_PROPS)]

        vplot = ax.violinplot(bin_values, positions=positions, widths=widths, showmedians=False)
        for patch in vplot["bodies"]:
            patch.set_facecolor(color)
        for partname in ("cbars", "cmins", "cmaxes"):
            vplot[partname].set_edgecolor("black")
            vplot[partname].set_linewidth(1)

        if with_aucs:
            uncertainty_error_lines_and_aucs = self._get_uncertainty_by_error_lines_and_aucs(
                df, df_grouped_by_error_bins
            )
            mean_line_handle = ax.plot(
                uncertainty_error_lines_and_aucs["mean_line_positions"],
                uncertainty_error_lines_and_aucs["mean_line_values"],
                linestyle="solid",
                color=color,
                label=(
                    "Mean Uncertainty Threshold by Binned ({} Years) Error; AUC={:.2f}".format(
                        bin_width,
                        uncertainty_error_lines_and_aucs["mean_line_auc"],
                    )
                ),
            )
            min_line_handle = ax.plot(
                uncertainty_error_lines_and_aucs["min_line_positions"],
                uncertainty_error_lines_and_aucs["min_line_values"],
                linestyle="dashed",
                color=color,
                label="Min Uncertainty Threshold by Error; AUC={:.2f}".format(
                    uncertainty_error_lines_and_aucs["min_line_auc"],
                ),
            )
            legend_handles.extend([*mean_line_handle, *min_line_handle])

        # Mean Diamonds
        for pos, mean in zip(positions, means):
            ax.plot(pos, mean, **MEAN_POINT_PROPS)

        ax.legend(handles=legend_handles)
        ax.set_ylim(bottom=0)
        ax.set_xlabel(f"Error / (Years) [Bin Width = {bin_width} Years]")
        ax.set_ylabel("Uncertainty")
        ax.set_xticks(error_bins)
        self._save_figure(
            fig,
            "uncertainty_by_error_with_aucs" if with_aucs else "uncertainty_by_error",
        )

    def _get_uncertainty_by_error_lines_and_aucs(
        self,
        df: pd.DataFrame,
        df_grouped_by_error_bins: pd.DataFrame,
    ) -> dict:
        mean_line_positions = []
        mean_line_values = []
        for bin, row in df_grouped_by_error_bins.sort_index(ascending=False).iterrows():
            uncertainty = row["uncertainty"]["mean"]
            if len(mean_line_positions) == 0:
                mean_line_positions.append(bin.right)
                mean_line_values.append(uncertainty)
            elif uncertainty < mean_line_values[-1]:
                mean_line_positions.extend([bin.right, bin.right])
                mean_line_values.extend([mean_line_values[-1], uncertainty])
        if mean_line_positions[0] != 0:
            mean_line_positions.append(0)
            mean_line_values.append(mean_line_values[-1])

        min_line_positions = []
        min_line_values = []
        for idx, row in df.sort_values("error", ascending=False).iterrows():
            uncertainty = row["uncertainty"]
            error = row["error"]
            if len(min_line_positions) == 0:
                min_line_positions.append(error)
                min_line_values.append(uncertainty)
            elif uncertainty < min_line_values[-1]:
                min_line_positions.extend([error, error])
                min_line_values.extend([min_line_values[-1], uncertainty])
        if min_line_positions[0] != 0:
            min_line_positions.append(0)
            min_line_values.append(min_line_values[-1])

        # Mean AUC
        mean_line_values_auc = np.subtract(mean_line_values, mean_line_values[-1])
        if mean_line_values_auc[0] > 0:
            mean_line_values_auc = np.divide(mean_line_values_auc, mean_line_values_auc[0])
        mean_line_values_auc = list(mean_line_values_auc)
        mean_line_positions_auc = list(np.divide(mean_line_positions, mean_line_positions[0]))
        mean_line_auc = 0
        while len(mean_line_positions_auc) > 0:
            width = abs(mean_line_positions_auc.pop() - mean_line_positions_auc.pop())
            height = mean_line_values_auc.pop()
            mean_line_auc += width * height
            mean_line_values_auc.pop()  # remove second entry for same height

        # Min AUC
        min_line_values_auc = np.subtract(min_line_values, min_line_values[-1])
        if min_line_values_auc[0] > 0:
            min_line_values_auc = np.divide(min_line_values_auc, min_line_values_auc[0])
        min_line_values_auc = list(min_line_values_auc)
        min_line_positions_auc = list(np.divide(min_line_positions, min_line_positions[0]))
        min_line_auc = 0
        while len(min_line_positions_auc) > 0:
            width = abs(min_line_positions_auc.pop() - min_line_positions_auc.pop())
            height = min_line_values_auc.pop()
            min_line_auc += width * height
            min_line_values_auc.pop()  # remove second entry for same height

        return dict(
            mean_line_positions=mean_line_positions,
            mean_line_values=mean_line_values,
            mean_line_auc=mean_line_auc,
            min_line_positions=min_line_positions,
            min_line_values=min_line_values,
            min_line_auc=min_line_auc,
        )

    def plot_error_by_age(self, eval_cfg_name: str) -> None:
        df = self.eval_runs_data[eval_cfg_name]["prediction_log"]
        target_min = int(np.floor(df["target"].min()))
        target_max = int(np.ceil(df["target"].max()))
        age_bins = np.linspace(start=target_min, stop=target_max, num=target_max - target_min + 1)
        age_bins_series = pd.cut(df["target"], bins=age_bins)
        df_binned_by_age = df.groupby(age_bins_series).agg({"error": [list, "mean"]}).dropna()

        bin_values = []
        positions = []
        widths = []
        means = []
        for bin, val in df_binned_by_age.iterrows():
            bin_values.append(val["error"]["list"])
            means.append(val["error"]["mean"])
            width = bin.right - bin.left
            positions.append((width / 2) + bin.left)
            widths.append(width)

        fig, ax = self._get_figure(
            title="Error by Age",
            derive_suptitle_from_cfg=eval_cfg_name,
        )

        vplot = ax.violinplot(bin_values, positions=positions, widths=widths, showmedians=False)
        for patch in vplot["bodies"]:
            patch.set_facecolor(self.eval_runs_data[eval_cfg_name]["color"])
        for partname in ("cbars", "cmins", "cmaxes"):
            vplot[partname].set_edgecolor("black")
            vplot[partname].set_linewidth(1)

        for pos, mean in zip(positions, means):
            ax.plot(pos, mean, **MEAN_POINT_PROPS)

        ax.legend(handles=[Line2D([0], [0], **MEAN_LEGEND_ENTRY_PROPS)])
        ax.set_xlabel("Age / (Years) [Bin Width = 1 Year]")
        ax.set_ylabel("Error / (Years)")
        ax.set_xticks(age_bins)
        self._save_figure(fig, "error_by_age")

    def plot_prediction_vs_truth(self, eval_cfg_name: str) -> None:
        df = self.eval_runs_data[eval_cfg_name]["prediction_log"]
        target = df["target"].tolist()
        prediction = df["prediction"].tolist()

        fig, ax = self._get_figure(
            title="Age Prediction vs. Ground Truth",
            figsize=(10, 10),
            derive_suptitle_from_cfg=eval_cfg_name,
        )
        color = self.eval_runs_data[eval_cfg_name]["color"]
        ax.plot(target, prediction, ".", label="Predictions", color=color)
        ax.plot(target, target, "-", label="Ground Truth", color=TARGET_COLOR)
        ax.legend()
        ax.set_xlabel("Ground Truth Age / (Years)")
        ax.set_ylabel("Predicted Age / (Years)")
        self._save_figure(fig, "prediction_vs_truth")

    def plot_error_by_abstention_rate(
        self,
        eval_cfg_names: Optional[list[str]] = None,
        only_p95: bool = False,
    ) -> None:
        """
        Plot the absolute error and/or maximum p95 error against the abstention by uncertainty.
        Saves AUC and Abstention info/overview csvs, too.

        Args:
            eval_cfg_names: List of eval configs to include in the plot.
            only_p95: If `true`, no maximum absolute error will be plotted.
        """
        if not eval_cfg_names:
            eval_cfg_names = list(self.eval_runs_data.keys())

        is_comparison = len(eval_cfg_names) > 1
        fig, ax = self._get_figure(
            title="Error by Abstention Rate",
            suptitle=self.eval_runs_data[eval_cfg_names[0]]["data_display_name"],
        )
        legend_handles = []

        global_error_max = 0
        global_error_95p_max = 0

        error_by_abstention_aucs = pd.DataFrame(
            columns=[
                "eval_cfg_name",
                "error_auc",
                "error_auc_norm",
                "error_auc_norm_glob",
                "error_95p_auc",
                "error_95p_auc_norm",
                "error_95p_auc_norm_glob",
            ],
        ).set_index("eval_cfg_name")
        abstentions = pd.DataFrame(
            columns=[
                "eval_cfg_name",
                "max_abstention",
                "exact_abstention",
                "max_error",
                "max_error_95p",
                "uncertainty_threshold",
            ],
        ).set_index(keys=["eval_cfg_name", "max_abstention"])

        for eval_cfg_name in eval_cfg_names:
            df = self.eval_runs_data[eval_cfg_name]["prediction_log"]
            df = df.copy(deep=True).sort_values("uncertainty")
            df_rolling = df.set_index("uncertainty").rolling(len(df), min_periods=1)
            df["error_running_max"] = df_rolling["error"].max().values
            df["error_95p_running"] = df_rolling["error"].quantile(0.95).values
            df_rolling = df.set_index("uncertainty").rolling(len(df), min_periods=1)
            df["error_95p_running_max"] = df_rolling["error_95p_running"].max().values
            df["pos"] = np.arange(len(df))
            df["prediction_abstention_rate"] = 1 - (df["pos"] + 1) / len(df)

            color = self.eval_runs_data[eval_cfg_name]["color"]
            error_max = df["error_running_max"].tolist()
            error_95p_max = df["error_95p_running_max"].tolist()
            abstention_rate = df["prediction_abstention_rate"].tolist()

            global_error_max = max(global_error_max, np.max(error_max))
            global_error_95p_max = max(global_error_95p_max, np.max(error_95p_max))

            # Abstentention UQ Thresholds
            for max_abstention in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                row = df[df["prediction_abstention_rate"] <= max_abstention].iloc[0]
                idx = (eval_cfg_name, max_abstention)
                abstentions.loc[idx, "exact_abstention"] = row["prediction_abstention_rate"]
                abstentions.loc[idx, "max_error"] = row["error_running_max"]
                abstentions.loc[idx, "max_error_95p"] = row["error_95p_running_max"]
                abstentions.loc[idx, "uncertainty_threshold"] = row["uncertainty"]

            # AUC Calculation
            width = 1 / len(df)
            error_auc = sum([error * width for error in df["error_running_max"].tolist()])
            error_auc_norm = error_auc / df["error_running_max"].max()
            error_95p_auc = sum([error * width for error in df["error_95p_running_max"].tolist()])
            error_95p_auc_norm = error_95p_auc / df["error_95p_running_max"].max()
            error_by_abstention_aucs.loc[eval_cfg_name, "error_auc"] = error_auc
            error_by_abstention_aucs.loc[eval_cfg_name, "error_auc_norm"] = error_auc_norm
            error_by_abstention_aucs.loc[eval_cfg_name, "error_95p_auc"] = error_95p_auc
            error_by_abstention_aucs.loc[eval_cfg_name, "error_95p_auc_norm"] = error_95p_auc_norm

            if not only_p95:
                ax.plot(abstention_rate, error_max, color=color, linestyle="dotted")
            ax.plot(
                abstention_rate,
                error_95p_max,
                color=color,
            )

            legend_handles.append(
                Patch(
                    facecolor=color,
                    edgecolor="black",
                    label=self.eval_runs_data[eval_cfg_name]["display_name"],
                ),
            )

        # Calculate Global Normalized AUCs
        error_by_abstention_aucs["error_auc_norm_glob"] = (
            error_by_abstention_aucs["error_auc"] / global_error_max
        )
        error_by_abstention_aucs["error_95p_auc_norm_glob"] = (
            error_by_abstention_aucs["error_95p_auc"] / global_error_95p_max
        )

        if not only_p95:
            percentile_handles = [
                Line2D([0], [0], color="black", linestyle="dotted", label="Max Error"),
                Line2D([0], [0], color="black", linestyle="solid", label="95% Percentile Error"),
            ]
            ax.add_artist(
                ax.legend(
                    loc="lower left",
                    handles=percentile_handles,
                    title="Evaluation Method",
                )
            )
        if is_comparison:
            ax.legend(
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                handles=legend_handles,
                title="UQ Method",
            )

        ax.set_xlabel("Abstention Rate / (%)")
        ax.set_ylabel(f"Max (95th Percentile) Error / (Years)")
        filename = "error_by_abstention_rate"
        if only_p95:
            filename = f"{filename}_95p"
        if is_comparison > 1:
            filename = f"{filename}_comparison"
        self._save_figure(fig, filename)
        self._save_dataframe(
            error_by_abstention_aucs.sort_values("error_95p_auc"),
            "error_by_abstention_aucs",
        )
        self._save_dataframe(abstentions, "abstentions")

    def plot_calibration_curve(
        self,
        eval_cfg_names: Optional[list[str]] = None,
        comparison_plot: bool = False,
    ) -> None:
        """Calibration Curve - adapted from Deep Ensemble Paper.

        See: https://arxiv.org/abs/1612.01474

        Args:
            eval_cfg_names: A list of configuration names to include into the plot. Must be given,
                if plot is NOT a `comparison_plot` (default).
            comparison_plot: If the plots shows a single UQ method or a comparison of multiple ones.
        """

        if not eval_cfg_names:
            if not comparison_plot:
                raise ValueError("Only comparison plot allows empty `eval_cfg_names`!")
            eval_cfg_names = list(self.eval_runs_data.keys())

        fig, ax = self._get_figure(
            title=f"Calibration Curve {'Comparison ' if comparison_plot else ''}(WIP)",
            derive_suptitle_from_cfg=eval_cfg_names[0] if not comparison_plot else None,
            suptitle=(
                self.eval_runs_data[eval_cfg_names[0]]["data_display_name"]
                if comparison_plot
                else None
            ),
        )

        # Ideal Line
        ci_intervals = [i / 100 for i in range(0, 110, 10)]
        ax.plot(
            ci_intervals,
            ci_intervals,
            linestyle="--",
            color=TARGET_COLOR,
            label="Ideal Fraction",
        )

        for eval_cfg_name in eval_cfg_names:
            data = self.eval_runs_data[eval_cfg_name]
            df = data["prediction_log"].copy(deep=True)
            # Undo Age Transforms to operate on original metric (year, month, etc.)
            df = apply_df_age_transform(df, self.undo_age_to_year_transform)
            # Observations from the (test) set
            observations = df["target"].tolist()

            ci_shares = {k: [] for k in QUANTILE_SIGMA_ENV_SCALES.keys()}
            for i in range(len(df)):
                mean = df.iloc[i]["prediction"]
                var = df.iloc[i]["uncertainty"] ** 2

                sample_ci_shares = observation_share_per_prediction_interval(
                    mean, var, observations
                )
                for sample_ci_share, val in sample_ci_shares.items():
                    ci_shares[sample_ci_share].append(val)

            ci_share_means = []
            ci_share_lower_stds = []
            ci_share_upper_stds = []
            for ci_share, ci_share_vals in ci_shares.items():
                mean = np.mean(ci_share_vals)
                std = np.std(ci_share_vals)
                ci_share_means.append(mean)
                ci_share_lower_stds.append(max(mean - 3 * std, 0))
                ci_share_upper_stds.append(min(mean + 3 * std, 1))

            color_obs = data["color"]
            # 3*Std of Observed Fraction; only if not a comparison plot
            if not comparison_plot:
                ax.fill_between(
                    ci_intervals,
                    [0, *ci_share_lower_stds, 1],
                    [0, *ci_share_upper_stds, 1],
                    color=color_obs,
                    alpha=0.2,
                    label="Three Standard Deviations of Observed Fraction",
                )
            # Observed Fractions per Quantile
            ax.plot(
                ci_intervals,
                [0, *ci_share_means, 1],
                linestyle="-",
                marker=data["marker"],
                label=data["display_name"],
                color=color_obs,
            )

        ax.set_xlabel("Expected Fraction")
        ax.set_ylabel("Observed Fraction")
        ax.legend()
        name = "calibration_curve_comparison" if comparison_plot else "calibration_curve"
        self._save_figure(fig, name)

    def plot_correlation_comparison(
        self,
        eval_cfg_names: Optional[list[str]] = None,
        method: str = "pearson",
    ) -> None:
        corrs = []
        labels = []
        colors = []

        if not eval_cfg_names:
            eval_cfg_names = self.eval_runs_data.keys()

        for eval_cfg_name in eval_cfg_names:
            eval_run_data = self.eval_runs_data[eval_cfg_name]
            df = eval_run_data["prediction_log"]
            corr = float(df["uncertainty"].corr(df["error"], method=method))
            corrs.append(corr)
            labels.append(eval_run_data["display_name"])
            colors.append(eval_run_data["color"])

        fig, ax = self._get_figure(
            title=f"Uncertainty-Error {method.title()} Correlation Comparison",
            suptitle=eval_run_data["data_display_name"],  # Use data display name of last entry
        )
        ax.set_ylabel(f"({method.title()}) Correlation of Uncertainty and Error")
        ax.bar(labels, corrs, color=colors)
        ax.set_xticks(labels)
        ax.set_xticklabels(labels, rotation=30, ha="right", rotation_mode="anchor")
        ax.set_yticks(np.linspace(start=-1, stop=1, num=21))
        y_lim_min = max(-1, min(corrs) - 0.05)
        y_lim_max = min(1, max(corrs) + 0.05)
        ax.set_ylim((y_lim_min, y_lim_max))
        self._save_figure(fig, f"correlation_comparison_{method}")

    def plot_error_comparison(
        self,
        eval_cfg_names: Optional[list[str]] = None,
        plot_type: str = "boxplot",
    ) -> None:
        """Plot comparison of errors by different UQ method (Boxplot or Violin Plot)."""
        if not eval_cfg_names:
            eval_cfg_names = list(self.eval_runs_data.keys())

        errors = []
        labels = []
        colors = []
        legend_handles = [Line2D([0], [0], **MEAN_LEGEND_ENTRY_PROPS)]

        if self.baseline_model_error_df is not None:
            labels.append("Baseline")
            colors.append(BASELINE_MODEL_COLOR)
            errors.append(self.baseline_model_error_df["error"].tolist())

        for eval_cfg_name in eval_cfg_names:
            eval_run_data = self.eval_runs_data[eval_cfg_name]
            df = eval_run_data["prediction_log"]
            errors.append(df["error"].tolist())
            labels.append(eval_run_data["display_name"])
            colors.append(eval_run_data["color"])

        fig, ax = self._get_figure(
            title=f"Comparison of Error by (UQ) Method",
            suptitle=self.eval_runs_data[eval_cfg_names[0]]["data_display_name"],
        )

        if self.mean_predictor_model_error_df is not None:
            mp_line = ax.axvline(
                x=self.mean_predictor_model_error_df["error"].mean(),
                color="red",
                linestyle="dotted",
                label="Mean of Mean-Predictor",
            )
            legend_handles.append(mp_line)

        if plot_type == "boxplot":
            bplot = ax.boxplot(
                errors,
                labels=labels,
                patch_artist=True,
                vert=False,
                showmeans=True,
                meanline=False,
                meanprops=MEAN_POINT_PROPS,
                flierprops=BOXPLOT_FLIER_PROPS,
            )

            box_patches: Patch
            for box_patches, color in zip(bplot["boxes"], colors):
                color = to_rgba(color, alpha=0.5)
                box_patches.set_facecolor(color)
            median_lines: Line2D
            for median_lines in bplot["medians"]:
                median_lines.set_color("black")
            whisker_lines: Line2D
            for whisker_lines in [*bplot["whiskers"], *bplot["caps"]]:
                whisker_lines.set_dashes((2, 2))

        elif plot_type == "violin":
            positions = np.arange(1, len(labels) + 1)
            vplot = ax.violinplot(errors, showmeans=False, showmedians=True, vert=False)

            # Violin Facecolors
            for box_patches, color in zip(vplot["bodies"], colors):
                box_patches.set_facecolor(color)
            # Black Body Lines
            for partname in ("cbars", "cmins", "cmaxes", "cmedians"):
                vplot[partname].set_edgecolor("black")
                vplot[partname].set_linewidth(1)
            # Mean Markers
            for pos, err in zip(positions, errors):
                ax.plot(np.mean(err), pos, **MEAN_POINT_PROPS)

            ax.set_yticks(positions, labels=labels)
            ax.set_ylim(0.25, len(labels) + 0.75)

        else:
            raise ValueError(f"Invalid Plot-Type {plot_type}!")

        ax.legend(handles=legend_handles)
        ax.set_xlabel(f"Absolute Error / (Years)")
        self._save_figure(fig, f"error_comparison_{plot_type}")

    def plot_uncertainty_by_error_comparison(
        self,
        eval_cfg_names: Optional[list[str]] = None,
        bin_width: float = 0.5,
        bin_padding: float = 0.05,
    ) -> None:
        if not eval_cfg_names:
            eval_cfg_names = list(self.eval_runs_data.keys())

        meta_cfg = self.eval_runs_data[eval_cfg_names[0]]
        fig, ax = self._get_figure(
            figsize=(11, 7),
            title="Uncertainty by Error - Comparison",
            suptitle=meta_cfg["data_display_name"],
        )

        error_bins = self._get_error_bins(eval_cfg_names, bin_width=bin_width)
        legend_elements = []

        uq_method_distance = (bin_width - (2 * bin_padding)) / (len(eval_cfg_names) - 1)

        for i, eval_cfg_name in enumerate(eval_cfg_names):
            eval_cfg = self.eval_runs_data[eval_cfg_name]
            df = eval_cfg["prediction_log"]
            df_error_bins = pd.cut(df["error"], bins=error_bins)
            df_binned_by_error = df.groupby(df_error_bins).agg({"uncertainty": ["mean"]}).dropna()

            markerprops = dict(
                marker=eval_cfg["marker"],
                markeredgecolor="black",
                markerfacecolor=eval_cfg["color"],
                markersize=8,
            )

            # Plot Mean for UQ Method for Bins
            for bin, val in df_binned_by_error.iterrows():
                position = bin.left + bin_padding + i * uq_method_distance
                ax.plot(
                    position,
                    val["uncertainty"]["mean"],
                    **markerprops,
                )

            # Legend Entry
            legend_elements.append(
                Line2D([0], [0], color=(0, 0, 0, 0), label=eval_cfg["display_name"], **markerprops)
            )

        ax.set_ylim(bottom=0)
        ax.set_xticks(error_bins)
        ax.set_xlabel(f"Error / (Years) [Bin Width = {bin_width} Years]")
        ax.set_ylabel("Mean Uncertainty")
        ax.legend(
            handles=legend_elements,
            bbox_to_anchor=(1.0, 0.5),
            loc="center left",
            title="UQ Methods",
        )
        self._save_figure(fig, "uncertainty_by_error_comparison")

    def save_uncertainty_reorder_ranks_csv(
        self, eval_cfg_names: Optional[list[str]] = None
    ) -> None:
        if not eval_cfg_names:
            eval_cfg_names = list(self.eval_runs_data.keys())

        reorder_ranks = pd.DataFrame(
            columns=[
                "eval_cfg_name",
                "name",
                "sort_min_swaps_rank",
                "sort_avg_idx_distance",
                "sort_avg_idx_right_move",
                "sort_idx_right_move_count",
            ],
        ).set_index("eval_cfg_name")

        for eval_cfg_name in eval_cfg_names:
            eval_cfg = self.eval_runs_data[eval_cfg_name]
            df = eval_cfg["prediction_log"].sort_values("uncertainty")
            df["pos"] = np.arange(len(df))  # positions of samples by sorted uncertainty

            reorder_ranks.loc[eval_cfg_name, "name"] = eval_cfg["display_name"]
            reorder_ranks.loc[eval_cfg_name, "sort_min_swaps_rank"] = sort_min_swaps(
                df["error"].tolist()
            )
            reorder_ranks.loc[eval_cfg_name, "sort_avg_idx_distance"] = np.mean(
                [abs(i - pos) for i, pos in enumerate(df.sort_values("error")["pos"])]
            )
            reorder_ranks.loc[eval_cfg_name, "sort_avg_idx_right_move"] = np.mean(
                [max(0, (i - pos)) for i, pos in enumerate(df.sort_values("error")["pos"])]
            )
            reorder_ranks.loc[eval_cfg_name, "sort_idx_right_move_count"] = np.sum(
                [1 if (i - pos) > 0 else 0 for i, pos in enumerate(df.sort_values("error")["pos"])]
            )

        reorder_ranks["sort_avg_idx_distance_norm"] = reorder_ranks["sort_avg_idx_distance"] / (
            len(df) / 2
        )
        reorder_ranks["sort_avg_idx_right_move_norm"] = reorder_ranks["sort_avg_idx_right_move"] / (
            len(df) / 2
        )
        self._save_dataframe(reorder_ranks, "uncertainty_by_error_reorder_ranks")

    def save_uncertainty_by_error_aucs_csv(
        self,
        eval_cfg_names: Optional[list[str]] = None,
        bin_width: float = 0.5,
    ) -> None:
        if not eval_cfg_names:
            eval_cfg_names = list(self.eval_runs_data.keys())

        error_bins = self._get_error_bins(eval_cfg_names, bin_width=bin_width)
        aucs_df = pd.DataFrame(
            columns=[
                "eval_cfg_name",
                "name",
                "mean_line_auc",
                "min_line_auc",
            ],
        ).set_index("eval_cfg_name")

        for eval_cfg_name in eval_cfg_names:
            eval_cfg = self.eval_runs_data[eval_cfg_name]
            df = eval_cfg["prediction_log"]
            df_error_bins = pd.cut(df["error"], bins=error_bins)
            df_grouped_by_error_bins = (
                df.groupby(df_error_bins).agg({"uncertainty": ["mean"]}).dropna()
            )
            uncertainty_by_error_lines_and_aucs = self._get_uncertainty_by_error_lines_and_aucs(
                df, df_grouped_by_error_bins
            )
            aucs_df.loc[eval_cfg_name, "name"] = eval_cfg["display_name"]
            aucs_df.loc[eval_cfg_name, "mean_line_auc"] = uncertainty_by_error_lines_and_aucs[
                "mean_line_auc"
            ]
            aucs_df.loc[eval_cfg_name, "min_line_auc"] = uncertainty_by_error_lines_and_aucs[
                "min_line_auc"
            ]

        aucs_df["mean_to_half"] = abs(aucs_df["mean_line_auc"] - 0.5)
        aucs_df["min_to_half"] = abs(aucs_df["min_line_auc"] - 0.5)
        self._save_dataframe(aucs_df, "uncertainty_by_error_aucs")

    def plot_uncertainty_by_error_aucs_comparison(
        self,
        eval_cfg_names: Optional[list[str]] = None,
        plot_type: Literal["mean", "min", "mean_min"] = "mean_min",
        bin_width: float = 0.5,
    ) -> None:
        if not eval_cfg_names:
            eval_cfg_names = list(self.eval_runs_data.keys())

        meta_cfg = self.eval_runs_data[eval_cfg_names[0]]
        fig, ax = self._get_figure(
            figsize=(11, 7),
            title="Uncertainty by Error AUC - Comparison",
            suptitle=meta_cfg["data_display_name"],
        )

        error_bins = self._get_error_bins(eval_cfg_names, bin_width=bin_width)
        legend_handles = []
        mean_min_handles = []

        for i, eval_cfg_name in enumerate(eval_cfg_names):
            eval_cfg = self.eval_runs_data[eval_cfg_name]
            df = eval_cfg["prediction_log"]
            df_error_bins = pd.cut(df["error"], bins=error_bins)
            df_grouped_by_error_bins = (
                df.groupby(df_error_bins).agg({"uncertainty": ["mean"]}).dropna()
            )

            uncertainty_by_error_lines_and_aucs = self._get_uncertainty_by_error_lines_and_aucs(
                df, df_grouped_by_error_bins
            )
            if plot_type in ["mean", "mean_min"]:
                ax.plot(
                    uncertainty_by_error_lines_and_aucs["mean_line_positions"],
                    uncertainty_by_error_lines_and_aucs["mean_line_values"],
                    linestyle="solid",
                    color=eval_cfg["color"],
                )
            if plot_type in ["min", "mean_min"]:
                ax.plot(
                    uncertainty_by_error_lines_and_aucs["min_line_positions"],
                    uncertainty_by_error_lines_and_aucs["min_line_values"],
                    linestyle="dashed",
                    color=eval_cfg["color"],
                )

            # Legend Entry
            legend_handles.append(
                Patch(
                    facecolor=eval_cfg["color"],
                    edgecolor="black",
                    label=eval_cfg["display_name"],
                ),
            )

        ax.set_ylim(bottom=0)
        ax.set_xticks(error_bins)
        ax.set_xlabel(f"Error / (Years)")
        ax.set_ylabel("Uncertainty")
        if plot_type in ["mean", "mean_min"]:
            mean_min_handles.append(
                Line2D(
                    [0],
                    [0],
                    color="black",
                    linestyle="solid",
                    label="Mean Uncertainty Threshold by Binned ({:.1f} Years) Error".format(
                        bin_width,
                    ),
                )
            )
        if plot_type in ["min", "mean_min"]:
            mean_min_handles.append(
                Line2D(
                    [0],
                    [0],
                    color="black",
                    linestyle="dotted",
                    label="Min Uncertainty Threshold",
                )
            )

        ax.add_artist(ax.legend(handles=mean_min_handles, title="Evaluation Method"))
        ax.legend(
            handles=legend_handles,
            bbox_to_anchor=(1.0, 0.5),
            loc="center left",
            title="UQ Methods",
        )
        self._save_figure(fig, f"uncertainty_by_error_auc_comparison_{plot_type}")

    def _get_error_bins(self, eval_cfg_names: list[str], bin_width: float = 0.5) -> np.ndarray:
        errors = []
        for eval_cfg_name in eval_cfg_names:
            errors.extend(self.eval_runs_data[eval_cfg_name]["prediction_log"]["error"].tolist())

        error_min = int(np.floor(np.min(errors)))
        error_max = int(np.ceil(np.max(errors)))
        error_bins = np.linspace(
            start=error_min,
            stop=error_max + bin_width,
            endpoint=False,
            num=int(((error_max - error_min) / bin_width) + 1),
        )
        return error_bins

    def plot_error_by_age_comparison(
        self,
        eval_cfg_names: Optional[list[str]] = None,
    ) -> None:
        if not eval_cfg_names:
            eval_cfg_names = list(self.eval_runs_data.keys())

        meta_cfg = self.eval_runs_data[eval_cfg_names[0]]
        fig, ax = self._get_figure(
            title="Error by Age - Comparison",
            suptitle=meta_cfg["data_display_name"],
            figsize=(11, 7),
        )

        target_min = int(np.floor(meta_cfg["prediction_log"]["target"].min()))
        target_max = int(np.ceil(meta_cfg["prediction_log"]["target"].max()))
        age_bins = np.linspace(start=target_min, stop=target_max, num=target_max - target_min + 1)
        legend_elements = []

        bin_padding = 0.05  # space between bin border and first/ last marker
        uq_method_distance = (1 - (2 * bin_padding)) / (len(eval_cfg_names) - 1)

        for i, eval_cfg_name in enumerate(eval_cfg_names):
            eval_cfg = self.eval_runs_data[eval_cfg_name]
            df = eval_cfg["prediction_log"]
            df_age_bins = pd.cut(df["target"], bins=age_bins)
            df_binned_by_target = df.groupby(df_age_bins).agg({"error": ["mean"]}).dropna()

            markerprops = dict(
                marker=eval_cfg["marker"],
                markeredgecolor="black",
                markerfacecolor=eval_cfg["color"],
                markersize=8,
            )

            # Plot Mean for UQ Method for Bins
            for bin, val in df_binned_by_target.iterrows():
                position = bin.left + bin_padding + i * uq_method_distance
                ax.plot(
                    position,
                    val["error"]["mean"],
                    **markerprops,
                )

            # Legend Entry
            legend_elements.append(
                Line2D([0], [0], color=(0, 0, 0, 0), label=eval_cfg["display_name"], **markerprops)
            )

        ax.set_ylim(bottom=0)
        ax.set_xticks(age_bins)
        ax.set_xlabel("Age / (Years) [Bin Width = 1 Year]")
        ax.set_ylabel("Mean Absolute Error")
        ax.legend(
            handles=legend_elements,
            bbox_to_anchor=(1.0, 0.5),
            loc="center left",
            title="UQ Methods",
        )
        self._save_figure(fig, "error_by_age_comparison")

    def save_error_uncertainty_stats(self, eval_cfg_names: Optional[list[str]] = None) -> None:
        if not eval_cfg_names:
            eval_cfg_names = list(self.eval_runs_data.keys())

        stats = pd.DataFrame(
            columns=[
                "eval_cfg_name",
                "name",
                "error_mean",
                "error_median",
                "error_std",
                "uncertainty_mean",
                "uncertainty_median",
                "uncertainty_std",
            ],
        ).set_index("eval_cfg_name")

        if self.mean_predictor_model_error_df is not None:
            stats.loc["mean_predictor", "name"] = "Mean-Predictor"
            stats.loc["mean_predictor", "error_mean"] = self.mean_predictor_model_error_df[
                "error"
            ].mean()
            stats.loc["mean_predictor", "error_median"] = self.mean_predictor_model_error_df[
                "error"
            ].median()
            stats.loc["mean_predictor", "error_std"] = self.mean_predictor_model_error_df[
                "error"
            ].std()
        if self.baseline_model_error_df is not None:
            stats.loc["baseline", "name"] = "Baseline"
            stats.loc["baseline", "error_mean"] = self.baseline_model_error_df["error"].mean()
            stats.loc["baseline", "error_median"] = self.baseline_model_error_df["error"].median()
            stats.loc["baseline", "error_std"] = self.baseline_model_error_df["error"].std()

        for eval_cfg_name in eval_cfg_names:
            df = self.eval_runs_data[eval_cfg_name]["prediction_log"]
            stats.loc[eval_cfg_name, "name"] = self.eval_runs_data[eval_cfg_name]["display_name"]
            stats.loc[eval_cfg_name, "error_mean"] = df["error"].mean()
            stats.loc[eval_cfg_name, "error_median"] = df["error"].median()
            stats.loc[eval_cfg_name, "error_std"] = df["error"].std()
            stats.loc[eval_cfg_name, "uncertainty_mean"] = df["uncertainty"].mean()
            stats.loc[eval_cfg_name, "uncertainty_median"] = df["uncertainty"].median()
            stats.loc[eval_cfg_name, "uncertainty_std"] = df["uncertainty"].std()

        self._save_dataframe(stats, "error_uncertainty_stats")

    def _get_figure(
        self,
        nrows: int = 1,
        ncols: int = 1,
        figsize: tuple[int, int] = (10, 7),
        dpi: int = 250,
        title: Optional[str] = None,
        suptitle: Optional[str] = None,
        derive_suptitle_from_cfg: Optional[str] = None,
    ):
        if suptitle and derive_suptitle_from_cfg:
            raise ValueError("Either suptitle given, or derive_suptitle - not both!")
        if title and (nrows > 1 or ncols > 1):
            raise ValueError("Title only allowed for single Axes Plot!")

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi)
        if title:
            assert isinstance(ax, Axes)
            ax.set_title(title)
        if derive_suptitle_from_cfg:
            suptitle = self._get_suptitle(derive_suptitle_from_cfg)
        if suptitle:
            fig.suptitle(suptitle)
        return fig, ax

    def _save_figure(self, fig: Figure, name: str, tight_layout: bool = True) -> None:
        """Save Given Figure."""
        os.makedirs(self.img_save_dir, exist_ok=True)
        filepath = self._get_save_filepath(name)
        if tight_layout:
            fig.tight_layout()
        fig.savefig(filepath, bbox_inches="tight")
        plt.close()

    def _get_suptitle(self, eval_cfg_name: str) -> str:
        suptitle = self.eval_runs_data[eval_cfg_name]["display_name"]
        data_name = self.eval_runs_data[eval_cfg_name]["data_display_name"]
        if data_name:
            suptitle = f"{data_name} - {suptitle}"
        return suptitle

    def _save_dataframe(self, df: pd.DataFrame, name: str) -> None:
        filepath = self._get_save_filepath(name, "csv")
        df.to_csv(filepath)

    def _get_save_filepath(self, name: str, extension: Optional[str] = None) -> str:
        """Generate Filename based on settings (prepend-str, with timestamp)."""
        if extension is None:
            extension = self.img_ext

        filename = f"{name}.{extension}"
        if self.img_with_timestamp:
            filename = f"{self.ts}_{filename}"
        if self.img_prepend_str:
            filename = f"{self.img_prepend_str}_{filename}"
        return os.path.join(self.img_save_dir, filename)
