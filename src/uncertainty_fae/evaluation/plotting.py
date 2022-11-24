import os
from datetime import datetime
from typing import Optional, TypedDict
from matplotlib.figure import Figure

import matplotlib.pyplot as plt
import numpy as np
import pandas
from pandas import DataFrame

from uncertainty_fae.evaluation.calibration import QUANTILE_SIGMA_ENV_SCALES, observation_share_per_prediction_interval

TARGET_COLOR = 'green'
"""Color to use in plots for the target (ground truth)."""


class EvalRunData(TypedDict):
    """Simple Wrapper for results data as required by the `EvalPlotGenerator`."""

    display_name: str
    data_display_name: Optional[str]
    prediction_log: DataFrame
    distinct_prediction_log: Optional[DataFrame]
    color: str


class EvalPlotGenerator():

    def __init__(
        self,
        eval_runs_data: dict[str, EvalRunData],
        img_save_dir: str,
        img_ext: str = 'png',
        plt_style: str = 'seaborn',
        show_interactive_plots = False,
        img_prepend_str: str = '',
        img_with_timestamp: bool = False,
    ) -> None:
        self.eval_runs_data = eval_runs_data
        self.img_save_dir = img_save_dir
        self.img_ext = img_ext
        self.show_interactive_plots = show_interactive_plots
        self.img_prepend_str = img_prepend_str
        self.img_with_timestamp = img_with_timestamp
        self.ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        plt.style.use(plt_style)

    def plot_bonage_distribution(self, eval_cfg_name: str, bins=25) -> None:
        df = self.eval_runs_data[eval_cfg_name]['prediction_log']
        p_color = self.eval_runs_data[eval_cfg_name]['color']
        self._init_figure(
            title='Distribution of Predicted and True Bone Age',
            derive_suptitle_from_cfg=eval_cfg_name,
        )
        plt.xlabel('(Predicted) Bone Age')
        plt.hist(
            [df['prediction'].tolist(), df['target'].tolist()],
            label=['Predicted Bone Age', 'True Bone Age'],
            color=[p_color, TARGET_COLOR],
            bins=bins,
        )
        plt.legend()
        self._save_and_show_plt('bonage_distribution')

    def plot_uncertainty_by_boneage(self, eval_cfg_name: str, bins=15) -> None:
        df = self.eval_runs_data[eval_cfg_name]['prediction_log']
        bins_target = pandas.cut(df['target'], bins=bins)
        df_binned_by_boneage = df.groupby(bins_target).agg({
            'error': [list, 'mean'],
            'uncertainty': [list, 'mean'],
            'target': 'mean',
        })

        bin_values = []
        positions = []
        widths = []
        ticks = []
        tick_labels = []
        for i, bin in enumerate(df_binned_by_boneage.iterrows()):
            bin_values.append(bin[1]['uncertainty']['list'])
            width = bin[0].right - bin[0].left
            position = (width / 2) + bin[0].left
            positions.append(position)
            widths.append(width)

            if i == 0:
                ticks.append(bin[0].left)
                tick_labels.append(round(bin[0].left, 1))
            ticks.append(bin[0].right)
            tick_labels.append(round(bin[0].right, 1))

        self._init_figure(
            title='Uncertainty by Bone Age',
            derive_suptitle_from_cfg=eval_cfg_name,
        )
        plt.xlabel('Bone Age')
        plt.ylabel('Uncertainty')
        vplot = plt.violinplot(bin_values, positions=positions, widths=widths, showmedians=True)
        for patch in vplot['bodies']:
            patch.set_facecolor(self.eval_runs_data[eval_cfg_name]['color'])
        # Make all the violin statistics marks red:
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
            vplot[partname].set_edgecolor('black')
            vplot[partname].set_linewidth(1)
        plt.xticks(ticks, tick_labels, rotation=45)
        self._save_and_show_plt('uncertainty_by_boneage')

    def plot_uncertainty_by_abs_error(self, eval_cfg_name: str, bins=15) -> None:
        df = self.eval_runs_data[eval_cfg_name]['prediction_log']
        bins_abs_error = pandas.cut(df['error'], bins=bins)
        df_binned_by_abs_error = df.groupby(bins_abs_error).agg(
            {'uncertainty': [list, 'mean']}).dropna()

        bin_values = []
        positions = []
        widths = []
        ticks = []
        tick_labels = []
        for i, bin in enumerate(df_binned_by_abs_error.iterrows()):
            bin_values.append(bin[1]['uncertainty']['list'])
            width = bin[0].right - bin[0].left
            positions.append((width / 2) + bin[0].left)
            widths.append(width)

            if i == 0:
                ticks.append(bin[0].left)
                tick_labels.append(round(bin[0].left, 1))
            ticks.append(bin[0].right)
            tick_labels.append(round(bin[0].right, 1))

        self._init_figure(
            title='Uncertainty by Absolute Error',
            derive_suptitle_from_cfg=eval_cfg_name,
        )
        plt.xlabel('Absolute Error')
        plt.ylabel('Uncertainty')
        vplot = plt.violinplot(bin_values, positions=positions, widths=widths, showmedians=True)
        for patch in vplot['bodies']:
            patch.set_facecolor(self.eval_runs_data[eval_cfg_name]['color'])
        # Make all the violin statistics marks red:
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
            vplot[partname].set_edgecolor('black')
            vplot[partname].set_linewidth(1)
        plt.xticks(ticks, tick_labels, rotation=45)
        self._save_and_show_plt('uncertainty_by_abs_error')

    def plot_abs_error_by_boneage(self, eval_cfg_name: str) -> None:
        df = self.eval_runs_data[eval_cfg_name]['prediction_log']
        bins_boneage = pandas.cut(df['target'], bins=15)
        df_binned_by_boneage = df.groupby(bins_boneage).agg(
            {'error': [list, 'mean']}).dropna()

        bin_values = []
        positions = []
        widths = []
        for bin in df_binned_by_boneage.iterrows():
            bin_values.append(bin[1]['error']['list'])
            width = bin[0].right - bin[0].left
            positions.append((width / 2) + bin[0].left)
            widths.append(width)

        self._init_figure(
            title='Absolute Error by Bone Age',
            derive_suptitle_from_cfg=eval_cfg_name,
        )
        plt.xlabel('Bone Age')
        plt.ylabel('Absolute Error')
        vplot = plt.violinplot(bin_values, positions=positions, widths=widths, showmedians=True)
        for patch in vplot['bodies']:
            patch.set_facecolor(self.eval_runs_data[eval_cfg_name]['color'])
        # Make all the violin statistics marks red:
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
            vplot[partname].set_edgecolor('black')
            vplot[partname].set_linewidth(1)
        self._save_and_show_plt('abs_error_by_boneage')

    def plot_prediction_vs_truth(self, eval_cfg_name: str) -> None:
        df = self.eval_runs_data[eval_cfg_name]['prediction_log']
        target = df['target'].tolist()
        prediction = df['prediction'].tolist()

        self._init_figure(
            title='Boneage Prediction vs. Ground Truth',
            figsize=(10, 10),
            derive_suptitle_from_cfg=eval_cfg_name,
        )
        color = self.eval_runs_data[eval_cfg_name]['color']
        plt.plot(target, prediction, '.', label='predictions', color=color)
        plt.plot(target, target, '-', label='actual', color=TARGET_COLOR)
        plt.legend()
        plt.xlabel('Boneage (Ground Truth)')
        plt.ylabel('Predicted Boneage')
        self._save_and_show_plt('prediction_vs_truth')

    def plot_tolerated_uncertainty_abs_error(self, eval_cfg_name: str) -> None:
        df = self.eval_runs_data[eval_cfg_name]['prediction_log']
        df = df.copy(True).sort_values('uncertainty')
        df_rolling = df.set_index('uncertainty').rolling(len(df), min_periods=1)
        df['abs_error_running_avg'] = df_rolling['error'].mean().values
        df['abs_error_running_max'] = df_rolling['error'].max().values
        df['abs_error_running_95_percentile'] = df_rolling['error'].quantile(0.95).values
        df['pos'] = np.arange(len(df))
        df['prediction_abstention_rate'] = 1 - (df['pos'] + 1) / len(df)

        uncertainties = df['uncertainty'].tolist()
        abs_error_avg = df['abs_error_running_avg'].tolist()
        abs_error_max = df['abs_error_running_max'].tolist()
        abs_error_95_percentile = df['abs_error_running_95_percentile'].tolist()
        abstention_rate = df['prediction_abstention_rate'].tolist()

        self._init_figure()
        fig, ax = plt.subplots()
        fig.suptitle(self._get_suptitle(eval_cfg_name))
        ax.set_title('Tolerated Uncertainty')
        p1 = ax.plot(uncertainties, abs_error_avg, label='Avg Abs Error')
        p2 = ax.plot(uncertainties, abs_error_max, label='Max Abs Error')
        p3 = ax.plot(uncertainties, abs_error_95_percentile, label='95 Percentile Abs Error')
        ax2 = ax.twinx()
        p4 = ax2.plot(uncertainties, abstention_rate, label='Abstention Rate', color='red')
        lns = p1 + p2 + p3 + p4
        labels = [l.get_label() for l in lns]
        plt.legend(lns, labels, loc='center right')
        ax.set_xlabel('Tolerated Uncertainty (Threshold)')
        ax.set_ylabel('Absolute Error')
        ax2.set_ylabel('Percentage')
        self._save_and_show_plt('tolerated_uncertainty')

    def plot_abstention_rate_vs_abs_error(
        self,
        eval_cfg_names: Optional[list[str]] = None,
        only_95_percentile: bool = False,
    ) -> None:
        if not eval_cfg_names:
            eval_cfg_names = self.eval_runs_data.keys()

        fig = self._init_figure(
            title='Absolute Error by Abstention Rate',
            suptitle=self.eval_runs_data[list(eval_cfg_names)[0]]['data_display_name'],
        )

        for eval_cfg_name in eval_cfg_names:
            df = self.eval_runs_data[eval_cfg_name]['prediction_log']
            df = df.copy(True).sort_values('uncertainty')
            df_rolling = df.set_index('uncertainty').rolling(len(df), min_periods=1)
            df['abs_error_running_avg'] = df_rolling['error'].mean().values
            df['abs_error_running_max'] = df_rolling['error'].max().values
            df['abs_error_running_95_percentile'] = df_rolling['error'].quantile(0.95).values
            df_rolling = df.set_index('uncertainty').rolling(len(df), min_periods=1)
            df['abs_error_running_95_percentile_max'] = df_rolling['abs_error_running_95_percentile'].max().values
            df['pos'] = np.arange(len(df))
            df['prediction_abstention_rate'] = 1 - (df['pos'] + 1) / len(df)

            color = self.eval_runs_data[eval_cfg_name]['color']
            uncertainties = df['uncertainty'].tolist()
            abs_error_avg = df['abs_error_running_avg'].tolist()
            abs_error_max = df['abs_error_running_max'].tolist()
            abs_error_95_percentile = df['abs_error_running_95_percentile'].tolist()
            abs_error_95_percentile_max = df['abs_error_running_95_percentile_max'].tolist()
            abstention_rate = df['prediction_abstention_rate'].tolist()

            if not only_95_percentile:
                plt.plot(
                    abstention_rate,
                    abs_error_max,
                    label=f'{self.eval_runs_data[eval_cfg_name]["display_name"]} - Max',
                    color=color,
                    linestyle='dotted'
                )
            plt.plot(
                abstention_rate,
                abs_error_95_percentile_max,
                label=f'{self.eval_runs_data[eval_cfg_name]["display_name"]} - 95% Percentile',
                color=color,
            )

        if len(eval_cfg_names) >= 2:
            # plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lower left', mode='expand', ncol=2)
            plt.legend(loc='lower left')
        else:
            plt.legend(loc='upper right')
        plt.xlabel('Abstention Rate (%)')
        plt.ylabel('Absolute Error')
        filename = 'abstention_rate_vs_abs_error'
        if only_95_percentile:
            filename = f'{filename}_95_perct'
        self._save_and_show_plt(filename)

    def plot_reliability_de_calibration_diagram(self, eval_cfg_name: str) -> None:
        """Calibration/ Reliability Diagram - DEPRECATED"""

        df = self.eval_runs_data[eval_cfg_name]['prediction_log']
        df_distinct = self.eval_runs_data[eval_cfg_name]['distinct_prediction_log']
        # Some UQ Methods do not provide this
        if df_distinct is None:
            return

        df_distinct = df_distinct.groupby('index').agg({'prediction': list})
        ci_shares = {k: [] for k in QUANTILE_SIGMA_ENV_SCALES.keys()}
        for i in range(len(df_distinct)):
            mean = df.iloc[i]['prediction']
            var = df.iloc[i]['uncertainty']**2
            observations = df_distinct.iloc[i]['prediction']
            sample_ci_shares = observation_share_per_prediction_interval(mean, var, observations)
            for sample_ci_share, val in sample_ci_shares.items():
                ci_shares[sample_ci_share].append(val)

        ci_intervals = [i/100 for i in range(0, 110, 10)]
        ci_share_means = []
        ci_share_lower_stds = []
        ci_share_upper_stds = []
        for ci_share, ci_share_vals in ci_shares.items():
            mean = np.mean(ci_share_vals)
            std = np.std(ci_share_vals)
            ci_share_means.append(mean)
            ci_share_lower_stds.append(max(mean - 3 * std, 0))
            ci_share_upper_stds.append(min(mean + 3 * std, 1))

        self._init_figure(
            title='Calibration (WIP) - FROM DE PAPER',
            derive_suptitle_from_cfg=eval_cfg_name,
        )
        color_obs = self.eval_runs_data[eval_cfg_name]['color']
        # Actual Observations Std
        plt.fill_between(
            ci_intervals,
            [0, *ci_share_lower_stds, 1],
            [0, *ci_share_upper_stds, 1],
            color=color_obs,
            alpha=0.2,
            label='Three Standard Deviations of Observed Fraction',
        )
        # Ideal Line
        plt.plot(
            ci_intervals,
            ci_intervals,
            linestyle="--",
            color=TARGET_COLOR,
            label='Ideal Fraction',
        )
        # Actual Observations
        plt.plot(
            ci_intervals,
            [0, *ci_share_means, 1],
            '^-',
            label='Observed Fraction',
            color=color_obs,
        )
        plt.xlabel('Expected Fraction')
        plt.ylabel('Observed Fraction')
        plt.legend()
        self._save_and_show_plt('reliability_de_calibration_diagram')

    def plot_calibration_curve(self, eval_cfg_names: Optional[list[str]] = None) -> None:
        """Calibration Curve - adapted from Deep Ensemble Paper

        See: https://arxiv.org/abs/1612.01474

        Args:
            eval_cfg_names: A list of configuration names to include into the plot.
        """

        if not eval_cfg_names:
            eval_cfg_names = self.eval_runs_data.keys()
        comparison_plot = len(eval_cfg_names) > 1
        ci_intervals = [i/100 for i in range(0, 110, 10)]

        self._init_figure(
            title=f'Calibration Curve {"Comparison " if comparison_plot else ""}(WIP)',
            derive_suptitle_from_cfg=eval_cfg_names[0] if not comparison_plot else None,
            suptitle=(self.eval_runs_data[list(eval_cfg_names)[0]]['data_display_name']
                      if comparison_plot else None),
        )
        # Ideal Line
        plt.plot(
            ci_intervals,
            ci_intervals,
            linestyle="--",
            color=TARGET_COLOR,
            label='Ideal Fraction',
        )

        for eval_cfg_name in eval_cfg_names:
            data = self.eval_runs_data[eval_cfg_name]
            df = data['prediction_log']
            # Observations from the (test) set
            observations = df['target'].tolist()
            
            ci_shares = {k: [] for k in QUANTILE_SIGMA_ENV_SCALES.keys()}
            for i in range(len(df)):
                mean = df.iloc[i]['prediction']
                var = df.iloc[i]['uncertainty']**2
                # observations = df_distinct.iloc[i]['prediction']
                sample_ci_shares = observation_share_per_prediction_interval(mean, var, observations)
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

            color_obs = data['color']
            # Actual Observations Std; only if not a comparison plot
            if not comparison_plot:
                plt.fill_between(
                    ci_intervals,
                    [0, *ci_share_lower_stds, 1],
                    [0, *ci_share_upper_stds, 1],
                    color=color_obs,
                    alpha=0.2,
                    label='Three Standard Deviations of Observed Fraction',
                )
            # Actual Observations
            plt.plot(
                ci_intervals,
                [0, *ci_share_means, 1],
                '^-',
                label=f'Observed Fraction - {data["display_name"]}',
                color=color_obs,
            )
        plt.xlabel('Expected Fraction')
        plt.ylabel('Observed Fraction')
        plt.legend()
        name = 'calibration_curve_comparison' if comparison_plot else 'calibration_curve'
        self._save_and_show_plt(name)

    def plot_reliability_de_calibration_diagram_comparison(
        self,
        eval_cfg_names: Optional[list[str]] = None
    ) -> None:
        """Comparison of Calibration/ Reliability Diagrams - DEPRECATED"""

        if not eval_cfg_names:
            eval_cfg_names = self.eval_runs_data.keys()

        self._init_figure(
            title='Calibration Comparison (WIP) - FROM DE PAPER',
            suptitle=self.eval_runs_data[list(eval_cfg_names)[0]]['data_display_name'],
        )
        ci_intervals = [i/100 for i in range(0, 110, 10)]
        # Ideal Line
        plt.plot(
            ci_intervals,
            ci_intervals,
            linestyle="--",
            color=TARGET_COLOR,
            label='Ideal Fraction',
        )

        for eval_cfg_name in eval_cfg_names:
            df = self.eval_runs_data[eval_cfg_name]['prediction_log']
            df_distinct = self.eval_runs_data[eval_cfg_name]['distinct_prediction_log']
            # Some UQ Methods do not provide this
            if df_distinct is None:
                continue

            df_distinct = df_distinct.groupby('index').agg({'prediction': list})
            ci_shares = {k: [] for k in QUANTILE_SIGMA_ENV_SCALES.keys()}
            for i in range(len(df_distinct)):
                mean = df.iloc[i]['prediction']
                var = df.iloc[i]['uncertainty']**2
                observations = df_distinct.iloc[i]['prediction']
                sample_ci_shares = observation_share_per_prediction_interval(
                    mean,
                    var,
                    observations,
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

            color_obs = self.eval_runs_data[eval_cfg_name]['color']
            # Actual Observations
            plt.plot(
                ci_intervals,
                [0, *ci_share_means, 1],
                '^-',
                label=self.eval_runs_data[eval_cfg_name]['display_name'],
                color=color_obs,
            )
        plt.xlabel('Expected Fraction')
        plt.ylabel('Observed Fraction')
        plt.legend()
        self._save_and_show_plt('reliability_de_calibration_diagram_comparison')

    def plot_correlation_comparison(
        self,
        eval_cfg_names: Optional[list[str]] = None,
        method: str = 'pearson',
    ) -> None:
        corrs = []
        labels = []
        colors = []

        if not eval_cfg_names:
            eval_cfg_names = self.eval_runs_data.keys()

        for eval_cfg_name in eval_cfg_names:
            eval_run_data = self.eval_runs_data[eval_cfg_name]
            df = eval_run_data['prediction_log']
            corr = float(df['uncertainty'].corr(df['error'], method=method))
            corrs.append(corr)
            labels.append(eval_run_data['display_name'])
            colors.append(eval_run_data['color'])

        self._init_figure(
            title=f'Uncertainty-Error {method.title()} Correlation Comparison',
            suptitle=eval_run_data['data_display_name']  # Use data display name of last entry
        )
        plt.ylabel(f'({method.title()}) Correlation of Uncertainty and Error')
        plt.bar(labels, corrs, color=colors)
        plt.xticks(rotation = 30)
        plt.ylim((-1, 1))
        self._save_and_show_plt(f'correlation_comparison_{method}')

    def plot_abs_error_comparison(
        self,
        eval_cfg_names: Optional[list[str]] = None,
        plot_type: str = 'violin',
    ) -> None:
        if not eval_cfg_names:
            eval_cfg_names = self.eval_runs_data.keys()

        abs_errors = []
        labels = []
        colors = []

        self._init_figure(
            title=f'Comparison of Absolute Error by (UQ) Method',
            suptitle=self.eval_runs_data[list(eval_cfg_names)[0]]['data_display_name'],
        )

        for eval_cfg_name in eval_cfg_names:
            eval_run_data = self.eval_runs_data[eval_cfg_name]
            df = eval_run_data['prediction_log']
            abs_errors.append(df['error'].tolist())
            labels.append(eval_run_data['display_name'])
            colors.append(eval_run_data['color'])

        if plot_type == 'boxplot':
            bplot = plt.boxplot(abs_errors, labels=labels, patch_artist=True)
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
        elif plot_type == 'violin':
            vplot = plt.violinplot(abs_errors, showmedians=True)
            for patch, color in zip(vplot['bodies'], colors):
                patch.set_facecolor(color)
            # Make all the violin statistics marks red:
            for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
                vplot[partname].set_edgecolor('black')
                vplot[partname].set_linewidth(1)
            # plt.tick_params('x', direction='out', position='bottom')
            plt.xticks(np.arange(1, len(labels) + 1), labels=labels)
            plt.xlim(0.25, len(labels) + 0.75)
        else:
            raise ValueError(f'Invalid Plot-Type {plot_type}!')
        plt.ylabel(f'Absolute Error')
        plt.xticks(rotation = 30)
        self._save_and_show_plt(f'abs_error_comparison_{plot_type}')

    def plot_uncertainty_by_abs_error_comparison(
        self,
        eval_cfg_names: Optional[list[str]] = None,
        bins=20
    ) -> None:

        if not eval_cfg_names:
            eval_cfg_names = self.eval_runs_data.keys()

        self._init_figure(
            title='Uncertainty by Absolute Error - Comparison',
            suptitle=self.eval_runs_data[list(eval_cfg_names)[0]]['data_display_name'],
        )

        for eval_cfg_name in eval_cfg_names:
            df = self.eval_runs_data[eval_cfg_name]['prediction_log']
            bins_abs_error = pandas.cut(df['error'], bins=bins)
            df_binned_by_abs_error = df.groupby(bins_abs_error).agg(
                {'uncertainty': [list, 'mean']}).dropna()

            uncertainty_means = []
            positions = []
            for bin in df_binned_by_abs_error.iterrows():
                uncertainty_means.append(bin[1]['uncertainty']['mean'])
                width = bin[0].right - bin[0].left
                positions.append((width / 2) + bin[0].left)
            plt.plot(
                positions,
                uncertainty_means,
                '.:',
                color=self.eval_runs_data[eval_cfg_name]['color'],
                label=self.eval_runs_data[eval_cfg_name]['display_name'],
            )
        plt.xlabel('Absolute Error')
        plt.ylabel('Uncertainty (Average)')        
        plt.legend()
        self._save_and_show_plt('uncertainty_by_abs_error_comparison')

    def plot_abs_error_by_boneage_comparison(
        self,
        eval_cfg_names: Optional[list[str]] = None,
        bins=20
    ) -> None:

        if not eval_cfg_names:
            eval_cfg_names = self.eval_runs_data.keys()

        self._init_figure(
            title='Absolute Error by Bone Age - Comparison',
            suptitle=self.eval_runs_data[list(eval_cfg_names)[0]]['data_display_name'],
        )

        for eval_cfg_name in eval_cfg_names:
            df = self.eval_runs_data[eval_cfg_name]['prediction_log']
            bins_abs_error = pandas.cut(df['target'], bins=bins)
            df_binned_by_target = df.groupby(bins_abs_error).agg(
                {'error': [list, 'mean']}).dropna()

            error_means = []
            positions = []
            for bin in df_binned_by_target.iterrows():
                error_means.append(bin[1]['error']['mean'])
                width = bin[0].right - bin[0].left
                positions.append((width / 2) + bin[0].left)
            plt.plot(
                positions,
                error_means,
                '.:',
                # linestyle=':',
                # marker='.',
                color=self.eval_runs_data[eval_cfg_name]['color'],
                label=self.eval_runs_data[eval_cfg_name]['display_name'],
            )
        plt.xlabel('Bone Age')
        plt.ylabel('Absolute Error (Average)')        
        plt.legend()
        self._save_and_show_plt('abs_error_by_boneage_comparison')

    def _init_figure(
        self,
        figsize: tuple[int, int] = (10, 7),
        dpi: int = 250,
        title: Optional[str] = None,
        suptitle: Optional[str] = None,
        derive_suptitle_from_cfg: Optional[str] = None,
    ) -> Figure:
        if suptitle and derive_suptitle_from_cfg:
            raise ValueError('Either suptitle given, or derive_suptitle - not both!')

        fig = plt.figure(figsize=figsize, dpi=dpi)
        plt.tight_layout()

        if title:
            plt.title(title)
        if derive_suptitle_from_cfg:
            suptitle = self._get_suptitle(derive_suptitle_from_cfg)
        if suptitle:
            plt.suptitle(suptitle)

        return fig

    def _get_suptitle(self, eval_cfg_name: str) -> str:
        suptitle = self.eval_runs_data[eval_cfg_name]['display_name']
        data_name = self.eval_runs_data[eval_cfg_name]['data_display_name']
        if data_name:
            suptitle = f'{data_name} - {suptitle}'
        return suptitle

    def _save_and_show_plt(self, name: str) -> None:
        os.makedirs(self.img_save_dir, exist_ok=True)

        # Generate Filename
        filename = f'{name}.{self.img_ext}'
        if self.img_with_timestamp:
            filename = f'{self.ts}_{filename}'
        if self.img_prepend_str:
            filename = f'{self.img_prepend_str}_{filename}'

        # Save and (optionally) show figure
        plt.savefig(os.path.join(self.img_save_dir, filename))
        if self.show_interactive_plots:
            plt.show(block=False)
        else:
            plt.close()
