import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas
from pandas import DataFrame


class EvalPlotGenerator():
    def __init__(self, log_df: DataFrame, img_save_dir: str, img_ext: str = 'png',
                 plt_style: str = 'seaborn', show_interactive_plots=False,
                 img_prepend_str: str = '', img_with_timestamp=True) -> None:
        self.log_df = log_df
        self.img_save_dir = img_save_dir
        self.img_ext = img_ext
        self.show_interactive_plots = show_interactive_plots
        self.img_prepend_str = img_prepend_str
        self.img_with_timestamp = img_with_timestamp
        self.ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        plt.style.use(plt_style)

    def plot_bonage_distribution(self, bins=25):
        self._init_figure()
        plt.title('Distribution of Predicted and True Boneage')
        plt.xlabel('(Predicted) Boneage')
        plt.hist([self.log_df['predicted_boneage'].tolist(), self.log_df['boneage'].tolist()],
                 label=['Predicted Boneage', 'True Boneage'], bins=bins)
        plt.legend()
        self._save_and_show_plt('bonage_distribution')
    
    def plot_abs_error_uncertainty_scatter(self):
        self._init_figure()
        plt.scatter(self.log_df['abs_error'].tolist(), self.log_df['uncertainty'].tolist(), 1)
        plt.title('Absolute Error vs. Uncertainty (Standard Deviation)')
        plt.xlabel('Absolute Error')
        plt.ylabel(f'Uncertainty')
        self._save_and_show_plt('abs_error_uncertainty_scatter')

    def plot_uncertainty_by_boneage(self, bins=15):
        bins_boneage = pandas.cut(self.log_df['boneage'], bins=bins)
        df_binned_by_boneage = self.log_df.groupby(bins_boneage).agg(
            {'abs_error': [list, 'mean'], 'uncertainty': [list, 'mean'], 'boneage': 'mean'})

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

        self._init_figure()
        plt.xlabel('Boneage')
        plt.ylabel('Uncertainty')
        plt.title('Uncertainty by Boneage')
        plt.violinplot(bin_values, positions=positions, widths=widths, showmedians=True)
        plt.xticks(ticks, tick_labels, rotation=45)
        plt.tight_layout()
        self._save_and_show_plt('uncertainty_by_boneage')

    def plot_uncertainty_by_abs_error(self, bins=15):
        bins_abs_error = pandas.cut(self.log_df['abs_error'], bins=bins)
        df_binned_by_abs_error = self.log_df.groupby(bins_abs_error).agg(
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

        self._init_figure()
        plt.xlabel('Absolute Error')
        plt.ylabel('Uncertainty')
        plt.title('Uncertainty by Absolute Error')
        plt.violinplot(bin_values, positions=positions, widths=widths, showmedians=True)
        plt.xticks(ticks, tick_labels, rotation=45)
        plt.tight_layout()
        self._save_and_show_plt('uncertainty_by_abs_error')

    def plot_uncertainty_comparison(self, bins=20):
        bins_abs_error = pandas.cut(self.log_df['abs_error'], bins=bins)
        aggregation = {
            'boneage': [list, 'mean'],
            'uncertainty': [list, 'mean'],
            'uncertainty_17_83': [list, 'mean'],
            'uncertainty_5_95': [list, 'mean'],
            'abs_error': 'mean',
        }
        df_binned_by_abs_error = self.log_df.groupby(bins_abs_error).agg(aggregation).dropna()

        self._init_figure()
        plt.xlabel('Absolute Error')
        plt.ylabel('Uncertainty (Average)')
        plt.title('Average Uncertainty by Binned Absolute Error (Comparison of Different Methods)')
        for measure, color, label in [('uncertainty', 'green', 'Standard Deviation'),
                                      ('uncertainty_17_83', 'darkred', '17 / 83 Percentile Range'),
                                      ('uncertainty_5_95', 'orange', '5 / 95 Percentile Range')]:
            y = []
            xmin = []
            xmax = []
            for bin in df_binned_by_abs_error.iterrows():
                y.append(bin[1][measure]['mean'])
                xmin.append(bin[0].left)
                xmax.append(bin[0].right)
            plt.hlines(y, xmin, xmax, colors=color, label=label)
        
        plt.legend(loc='upper left')
        self._save_and_show_plt('uncertainty_by_abs_error_comparison')

    def plot_abs_error_by_boneage(self):
        bins_boneage = pandas.cut(self.log_df['boneage'], bins=15)
        df_binned_by_boneage = self.log_df.groupby(bins_boneage).agg(
            {'abs_error': [list, 'mean']}).dropna()

        bin_values = []
        positions = []
        widths = []
        for bin in df_binned_by_boneage.iterrows():
            bin_values.append(bin[1]['abs_error']['list'])
            width = bin[0].right - bin[0].left
            positions.append((width / 2) + bin[0].left)
            widths.append(width)
        self._init_figure()
        plt.xlabel('Boneage')
        plt.ylabel('Absolute Error')
        plt.title('Absolute Error by Boneage')
        plt.violinplot(bin_values, positions=positions, widths=widths, showmedians=True)
        self._save_and_show_plt('abs_error_by_boneage')

    def plot_prediction_vs_truth(self):
        boneage_truth = self.log_df['boneage'].tolist()
        boneage_pred = self.log_df['predicted_boneage'].tolist()

        self._init_figure(figsize=(10, 10), title='Boneage Prediction vs. Ground Truth')
        plt.plot(boneage_truth, boneage_pred, 'r.', label = 'predictions')
        plt.plot(boneage_truth, boneage_truth, 'b-', label = 'actual')
        plt.legend()
        plt.xlabel('Boneage (Ground Truth)')
        plt.ylabel('Predicted Boneage')
        self._save_and_show_plt('prediction_vs_truth')

    def plot_tolerated_uncertainty_abs_error(self):
        df = self.log_df.copy(True).sort_values('uncertainty')
        df_rolling = df.set_index('uncertainty').rolling(len(df), min_periods=1)
        df['abs_error_running_avg'] = df_rolling['abs_error'].mean().values
        df['abs_error_running_max'] = df_rolling['abs_error'].max().values
        df['abs_error_running_95_percentile'] = df_rolling['abs_error'].quantile(0.95).values
        df['pos'] = np.arange(len(df))
        df['prediction_abstention_rate'] = 1 - (df['pos'] + 1) / len(df)

        uncertainties = df['uncertainty'].tolist()
        abs_error_avg = df['abs_error_running_avg'].tolist()
        abs_error_max = df['abs_error_running_max'].tolist()
        abs_error_95_percentile = df['abs_error_running_95_percentile'].tolist()
        abstention_rate = df['prediction_abstention_rate'].tolist()

        self._init_figure()
        fig, ax = plt.subplots()
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

    def plot_abstention_rate_vs_abs_error(self):
        df = self.log_df.copy(True).sort_values('uncertainty')
        df_rolling = df.set_index('uncertainty').rolling(len(df), min_periods=1)
        df['abs_error_running_avg'] = df_rolling['abs_error'].mean().values
        df['abs_error_running_max'] = df_rolling['abs_error'].max().values
        df['abs_error_running_95_percentile'] = df_rolling['abs_error'].quantile(0.95).values
        df_rolling = df.set_index('uncertainty').rolling(len(df), min_periods=1)
        df['abs_error_running_95_percentile_max'] = df_rolling['abs_error_running_95_percentile'].max().values
        df['pos'] = np.arange(len(df))
        df['prediction_abstention_rate'] = 1 - (df['pos'] + 1) / len(df)

        uncertainties = df['uncertainty'].tolist()
        abs_error_avg = df['abs_error_running_avg'].tolist()
        abs_error_max = df['abs_error_running_max'].tolist()
        abs_error_95_percentile = df['abs_error_running_95_percentile'].tolist()
        abs_error_95_percentile_max = df['abs_error_running_95_percentile_max'].tolist()
        abstention_rate = df['prediction_abstention_rate'].tolist()

        self._init_figure(title='Absolute Error by Abstention Rate')
        plt.plot(abstention_rate, abs_error_max, label='Max Abs Error')
        plt.plot(abstention_rate, abs_error_95_percentile_max, label='95% Percentile Abs Error')
        plt.xlabel('Abstention Rate (%)')
        plt.ylabel('Absolute Error')
        plt.legend(loc='upper right')
        # fig, ax = plt.subplots()
        # p1 = ax.plot(uncertainties, abs_error_avg, label='Avg Abs Error')
        # p2 = ax.plot(uncertainties, abs_error_max, label='Max Abs Error')
        # p3 = ax.plot(uncertainties, abs_error_95_percentile, label='95 Percentile Abs Error')
        # ax2 = ax.twinx()
        # p4 = ax2.plot(uncertainties, abstention_rate, label='Abstention Rate', color='red')
        # lns = p1 + p2 + p3 + p4
        # labels = [l.get_label() for l in lns]
        # plt.legend(lns, labels, loc='center right')
        # ax.set_xlabel('Tolerated Uncertainty (Threshold)')
        # ax.set_ylabel('Absolute Error')
        # ax2.set_ylabel('Percentage')
        self._save_and_show_plt('abstention_rate_vs_abs_error')

    def plot_reliability_diagram(self):
        data = self.log_df.sample(frac=0.001)
        ci_share_means = []
        ci_intervals = [i/100 for i in range(10, 110, 10)]
        ci_share_lower_stds = []
        ci_share_upper_stds = []
        for ci_col in [f'ci_{i}' for i in range(10, 100, 10)]:
            mean = data[ci_col].mean()
            std = data[ci_col].std()
            ci_share_means.append(mean)
            ci_share_lower_stds.append(max(mean - 3 * std, 0))
            ci_share_upper_stds.append(min(mean + 3 * std, 1))
        
        self._init_figure(title='Calibration (WIP)')
        plt.fill_between(ci_intervals[:-1], [*ci_share_lower_stds], [*ci_share_upper_stds],
                         color='red', alpha=0.2, label='Three Standard Deviations of Observed Fraction')
        # Ideal Line
        plt.plot([.0, *ci_intervals], [.0, *ci_intervals], linestyle="--", color='blue', label='Ideal Fraction')
        plt.plot(ci_intervals, [*ci_share_means, 1], '^-r', label='Observed Fraction')
        plt.xlabel('Expected Fraction')
        plt.ylabel('Observed Fraction')
        plt.legend()
        self._save_and_show_plt('reliability_diagram_calibration')

    def _init_figure(self, figsize=(10, 7), dpi=250, title=None):
        plt.figure(figsize=figsize, dpi=dpi)
        plt.tight_layout()
        if title:
            plt.title(title)

    def _save_and_show_plt(self, name: str):
        filename = f'{name}.{self.img_ext}'
        if self.img_with_timestamp:
            filename = f'{self.ts}_{filename}'
        if self.img_prepend_str:
            filename = f'{self.img_prepend_str}_{filename}'
        plt.savefig(os.path.join(self.img_save_dir, filename))
        if self.show_interactive_plots:
            plt.show(block=False)
        else:
            plt.close()