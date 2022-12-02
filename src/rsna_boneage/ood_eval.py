import csv
import logging
import os
from typing import Callable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from rsna_boneage.data import RSNABoneageDataModule
from uncertainty_fae.evaluation import OutOfDomainEvaluator
from uncertainty_fae.evaluation.util import EvalRunData
from uncertainty_fae.model import UncertaintyAwareModel
from uncertainty_fae.util import EvalRunConfig, ModelProvider

logger = logging.getLogger(__name__)

HATCHES = ['xxx', '***', 'ooo']


class RSNABoneAgeOutOfDomainEvaluator(OutOfDomainEvaluator):

    def __init__(
        self,
        data_base_dir: str,
        plot_base_dir: str,
        eval_run_cfg: EvalRunConfig,
        age_transform: Callable[[pd.Series], pd.Series],
    ) -> None:
        super().__init__(data_base_dir, plot_base_dir, eval_run_cfg, age_transform)

        self.ood_datasets: dict = self.eval_run_cfg.ood_datasets['rsna_boneage']
        self.plot_dir = os.path.join(self.plot_base_dir, self.eval_run_cfg.start_time)

    @classmethod
    def get_evaluator(
        cls,
        data_base_dir: str,
        plot_base_dir: str,
        eval_run_cfg: EvalRunConfig,
        age_transform: Callable[[pd.Series], pd.Series],
    ) -> 'OutOfDomainEvaluator':
        evaluator = cls(data_base_dir, plot_base_dir, eval_run_cfg, age_transform)
        return evaluator

    def _get_pred_filepath(self, eval_cfg_name: str, ood_name: str) -> str:
        return os.path.join(self.data_base_dir, f'{eval_cfg_name}_{ood_name}_ood_preds.csv')

    def _load_pred_file(self, eval_cfg_name: str, ood_name: str) -> pd.DataFrame | None:
        """Load Prediction DataFrame and apply Age Transforms."""
        ood_pred_file = self._get_pred_filepath(eval_cfg_name, ood_name)
        if not os.path.isfile(ood_pred_file):
            logger.warning(
                'No OoD-Pred file for %s, %s (%s)', eval_cfg_name, ood_name, ood_pred_file
            )
            return None
        ood_preds = pd.read_csv(ood_pred_file)
        for col in ['prediction', 'uncertainty']:
            ood_preds[col] = self.age_transform(ood_preds[col])
        return ood_preds

    def ood_preds_avail(self, eval_cfg_name: str) -> bool:
        for ood_name in self.ood_datasets.keys():
            pred_file = self._get_pred_filepath(eval_cfg_name, ood_name)
            if not os.path.exists(pred_file):
                return False  # If single pred file missing: return False
        return True

    def generate_predictions(
        self,
        eval_cfg_name: str,
        model: UncertaintyAwareModel,
        model_provider: ModelProvider,
    ) -> None:
        for ood_name, ood_cfg in self.ood_datasets.items():
            logger.info('Next OOD Dataset: %s', ood_name)
            pred_file = self._get_pred_filepath(eval_cfg_name, ood_name)

            if os.path.exists(pred_file):
                logger.info('Skipping, as prediction file already available...')
                continue

            dm: RSNABoneageDataModule = model_provider.get_lightning_data_module(
                None,
                ood_cfg['annotations'],
                None,
                img_val_base_dir=ood_cfg['base_dir'],
                batch_size=self.eval_run_cfg.batch_size,
                num_workers=self.eval_run_cfg.dataloader_num_workers,
            )
            dm.setup('validate')  # We always use the validation dataset
            dm.dataset_val.annotations = dm.dataset_val.annotations[:10000]  # max ood samples
            dm.dataset_val.annotations['boneage'] = 0  # we don't need the target to make sense
            # use random sex per sample
            dm.dataset_val.annotations['male'] = np.random.randint(0, 2, size=len(dm.dataset_val))

            results = model.evaluate_dataset(dm.val_dataloader())
            score, predictions, targets, errors, uncertainties, metrics = results

            # Prediction and Uncertainty Stats
            os.makedirs(self.data_base_dir, exist_ok=True)
            with open(pred_file, 'w') as file:
                writer = csv.writer(file)
                writer.writerow(['index', 'prediction', 'uncertainty'])
                writer.writerows(
                    zip(range(len(predictions)), predictions.tolist(), uncertainties.tolist())
                )

    def generate_plots(self, eval_runs_data: dict[str, EvalRunData]) -> None:
        self._generate_uq_comparison_plot(eval_runs_data)
        self._generate_prediction_comparison_plot(eval_runs_data)

    def _generate_uq_comparison_plot(self, eval_runs_data: dict[str, EvalRunData]) -> None:
        violin_positions = []
        violin_labels = []
        violin_datas = []
        violin_colors = []
        violin_hatches = []
        violin_edge_color = []
        legend_elements = [
            Patch(facecolor='white', edgecolor='black', label='RSNA Bone Age'),
        ]

        v_pos = 0
        for eval_cfg_name, eval_run_data in eval_runs_data.items():
            v_pos += 1
            uq_name = eval_run_data['display_name']
            uq_preds = eval_run_data['prediction_log']
            color = eval_run_data['color']

            avail_hatches = HATCHES.copy()

            # Base Entry: UQ with "normal" dataset
            violin_positions.append(v_pos)
            violin_labels.append(uq_name)
            violin_datas.append(uq_preds['uncertainty'].tolist())
            violin_colors.append(color)
            violin_hatches.append(None)  # no hatch for baseline dataset
            violin_edge_color.append('black')

            for ood_name, ood_cfg in self.ood_datasets.items():
                ood_preds = self._load_pred_file(eval_cfg_name, ood_name)
                if ood_preds is None:
                    continue

                v_pos += 1
                violin_positions.append(v_pos)
                # violin_labels.append(f'{uq_name} - {ood_cfg["name"]}')
                violin_labels.append('')
                violin_datas.append(ood_preds['uncertainty'].tolist())
                violin_colors.append(color)
                hatch = avail_hatches.pop(0)
                violin_hatches.append(hatch)
                violin_edge_color.append('white')

                if len(legend_elements) < len(self.ood_datasets.items()) + 1:
                    patch = Patch(
                        facecolor='white', edgecolor='grey', hatch=hatch, label=ood_cfg['name'])
                    legend_elements.append(patch)
            v_pos += 1

        fig, ax = self._get_fig()
        vplots = ax.violinplot(violin_datas, violin_positions, showmeans=True, widths=1)
        v_patch: PolyCollection
        for v_patch, color, hatch in zip(vplots['bodies'], violin_colors, violin_hatches):
            v_patch.set_facecolor(color)
            if hatch:
                v_patch.set_hatch(hatch)
                v_patch.set_edgecolor(color)
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
            vplots[partname].set_edgecolor('black')
            vplots[partname].set_linewidth(1)
        ax.set_xticks(violin_positions)
        ax.set_xticklabels(violin_labels, rotation=15)
        ax.set_ylabel('Uncertainty')
        ax.legend(handles=legend_elements, handleheight=3, handlelength=4, loc='upper left')
        self._save_fig(fig, 'uq_comparison')

    def _generate_prediction_comparison_plot(self, eval_runs_data: dict[str, EvalRunData]) -> None:
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
            uq_name = eval_run_data['display_name']
            color = eval_run_data['color']

            avail_hatches = HATCHES.copy()

            for ood_name, ood_cfg in self.ood_datasets.items():
                ood_preds = self._load_pred_file(eval_cfg_name, ood_name)
                if ood_preds is None:
                    continue

                v_pos += 1
                violin_positions.append(v_pos)
                violin_labels.append(uq_name)
                violin_datas.append(ood_preds['prediction'].tolist())
                violin_colors.append(color)
                hatch = avail_hatches.pop(0)
                violin_hatches.append(hatch)
                violin_edge_color.append('white')

                if len(legend_elements) < len(self.ood_datasets.items()):
                    patch = Patch(
                        facecolor='white', edgecolor='grey', hatch=hatch, label=ood_cfg['name'])
                    legend_elements.append(patch)
            v_pos += 1

        fig, ax = self._get_fig()
        vplots = ax.violinplot(violin_datas, violin_positions, showmeans=True, widths=1)
        v_patch: PolyCollection
        for v_patch, color, hatch in zip(vplots['bodies'], violin_colors, violin_hatches):
            v_patch.set_facecolor(color)
            if hatch:
                v_patch.set_hatch(hatch)
                v_patch.set_edgecolor(color)
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
            vplots[partname].set_edgecolor('black')
            vplots[partname].set_linewidth(1)
        ax.set_xticks(violin_positions)
        ax.set_xticklabels(violin_labels, rotation=15)
        ax.set_ylabel('Prediction (Age in Years)')
        ax.legend(handles=legend_elements, handleheight=3, handlelength=4, loc='upper right')
        self._save_fig(fig, 'prediction_comparison')

    def _get_fig(self):
        return plt.subplots(figsize=(10, 7), dpi=250)

    def _save_fig(self, fig: Figure, name: str) -> None:
        os.makedirs(self.plot_dir, exist_ok=True)
        fig.savefig(os.path.join(self.plot_dir, f'{name}.png'))
