import csv
import logging
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch

from rsna_boneage.data import RSNABoneageDataModule
from uncertainty_fae.evaluation import OutOfDomainEvaluator
from uncertainty_fae.evaluation.util import EvalRunData
from uncertainty_fae.model import UncertaintyAwareModel
from uncertainty_fae.util import EvalRunConfig, ModelProvider

logger = logging.getLogger(__name__)


class RSNABoneAgeOutOfDomainEvaluator(OutOfDomainEvaluator):

    def __init__(self, base_dir: str, eval_run_cfg: EvalRunConfig) -> None:
        super().__init__(base_dir, eval_run_cfg)

        self.ood_datasets: dict = self.eval_run_cfg.ood_datasets['rsna_boneage']

    @classmethod
    def get_evaluator(cls, base_dir: str, eval_run_cfg: EvalRunConfig) -> 'OutOfDomainEvaluator':
        evaluator = cls(base_dir, eval_run_cfg)
        return evaluator

    def _get_pred_filepath(self, eval_cfg_name: str, ood_name: str) -> str:
        return os.path.join(self.base_dir, f'{eval_cfg_name}_{ood_name}_ood_preds.csv')

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
            os.makedirs(self.base_dir, exist_ok=True)
            with open(pred_file, 'w') as file:
                writer = csv.writer(file)
                writer.writerow(['index', 'prediction', 'uncertainty'])
                writer.writerows(
                    zip(range(len(predictions)), predictions.tolist(), uncertainties.tolist())
                )

    def generate_plots(self, eval_runs_data: dict[str, EvalRunData]) -> None:
        self._generate_uq_comparison_plot(eval_runs_data)

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

            avail_hatches = ['xx', 'oo', '**']

            # Base Entry: UQ with "normal" dataset
            violin_positions.append(v_pos)
            violin_labels.append(uq_name)
            violin_datas.append(uq_preds['uncertainty'].tolist())
            violin_colors.append(color)
            violin_hatches.append(None)  # no hatch for baseline dataset
            violin_edge_color.append('black')

            for ood_name, ood_cfg in self.ood_datasets.items():
                ood_pred_file = self._get_pred_filepath(eval_cfg_name, ood_name)
                if not os.path.isfile(ood_pred_file):
                    logger.warning('No OOD-Pred file for %s, %s (%s)',
                                   eval_cfg_name, ood_name, ood_pred_file)
                    continue
                v_pos += 1

                ood_preds = pd.read_csv(ood_pred_file)
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

        fig, ax = plt.subplots()
        fig.set_size_inches(10, 6)
        vplots = ax.violinplot(violin_datas, violin_positions, showmeans=True)
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
        fig.savefig(os.path.join(self.base_dir, 'uq_comparison.png'))
