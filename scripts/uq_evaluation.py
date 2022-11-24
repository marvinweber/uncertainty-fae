import logging
import os

import pandas as pd
import yaml

from rsna_boneage.model_provider import RSNAModelProvider
from uncertainty_fae.evaluation import EvalPlotGenerator
from uncertainty_fae.evaluation.plotting import EvalRunData
from uncertainty_fae.evaluation.util import (evaluation_predictions_available,
                                             generate_evaluation_predictions)
from uncertainty_fae.model import UncertaintyAwareModel
from uncertainty_fae.util import EvalRunConfig, parse_cli_args
from uncertainty_fae.util.model_provider import ModelProvider

logger = logging.getLogger('UNCERTAINTY_FAE_EVALUATION')

PROVIDER_MAPPING: dict[str, ModelProvider] = {
    'rsna_boneage': RSNAModelProvider,
}

DATASET_NAMES = {
    'rsna_boneage': 'RSNA Bone Age',
    'clavicle_ct': 'Clavicle CT',
}

def evaluation_main(eval_run_cfg: EvalRunConfig) -> None:
    if len(eval_run_cfg.eval_only) == 0:
        logger.error('No Evaluation Configs could be found!')
        return
    else:
        logger.info('Running Evaluations for: %s', ', '.join(eval_run_cfg.eval_only))

    eval_runs_data: dict[str, EvalRunData] = {}

    for eval_cfg_name, eval_cfg in eval_run_cfg.eval_configuration.items():
        if eval_cfg_name not in eval_run_cfg.eval_only:
            logger.debug('Skipping config %s', eval_cfg_name)
            continue
        logger.info('NEXT EVALUATION --- %s (%s) for model "%s"',
                    eval_cfg_name, eval_cfg['name'], eval_cfg['model'])

        eval_cfg_name_base_dir = os.path.join(
            eval_run_cfg.eval_dir,
            eval_run_cfg.eval_version_name,
            eval_cfg_name,
            eval_run_cfg.dataset_type,
        )
        logger.info('Eval Log-Dir: %s', eval_cfg_name_base_dir)

        # GENERATING PREDICTIONS
        avail, *eval_files = evaluation_predictions_available(eval_cfg_name_base_dir)
        if not avail:
            model, dm = eval_run_cfg.get_model_and_datamodule(
                PROVIDER_MAPPING,
                eval_cfg['model'],
                model_checkpoint=eval_cfg['checkpoint'],
                eval_cfg_name=eval_cfg_name,
            )
            assert isinstance(model, UncertaintyAwareModel)
            dm.setup('test')
            model.set_dataloaders(
                train_dataloader=dm.train_dataloader(),
                val_dataloader=dm.val_dataloader(),
            )
            dataloader = eval_run_cfg.get_eval_dataloader(dm)
            eval_files = generate_evaluation_predictions(eval_cfg_name_base_dir, model, dataloader)
        else:
            logger.info('SKIPPING PREDICTIONS, as already available!')
        eval_result_file, eval_predictions_file, eval_distinct_predictions_file = eval_files

        data_type = eval_run_cfg.model_configurations[eval_cfg['model']]['data']
        prediction_log = pd.read_csv(eval_predictions_file)
        eval_runs_data[eval_cfg_name] = {
            'display_name': eval_cfg['name'],
            'data_display_name': DATASET_NAMES[data_type],
            'prediction_log': prediction_log,
            'distinct_prediction_log': (pd.read_csv(eval_distinct_predictions_file)
                                        if eval_distinct_predictions_file else None),
            'color': eval_cfg['color'] if 'color' in eval_cfg else 'black',
        }

        if eval_run_cfg.only_combined_plots:
            continue

        logger.info('Evaluating single model...')

        # SAVE EXTENDED EVAL STATS
        stats = {}
        for metric in ['prediction', 'uncertainty', 'error']:
            stats[f'{metric}_min'] = float(prediction_log[metric].min())
            stats[f'{metric}_max'] = float(prediction_log[metric].max())
            stats[f'{metric}_mean'] = float(prediction_log[metric].mean())
            stats[f'{metric}_std'] = float(prediction_log[metric].std())
            stats[f'{metric}_median'] = float(prediction_log[metric].median())
        with open(os.path.join(eval_cfg_name_base_dir, 'eval_metrics.yml'), 'w') as file:
            yaml.dump(stats, file)
        # Correlation Matrices
        prediction_log.corr(method='pearson').to_csv(
            os.path.join(eval_cfg_name_base_dir, 'corr_pearson.csv'))
        prediction_log.corr(method='kendall').to_csv(
            os.path.join(eval_cfg_name_base_dir, 'corr_kendall.csv'))
        prediction_log.corr(method='spearman').to_csv(
            os.path.join(eval_cfg_name_base_dir, 'corr_spearman.csv'))

        # CREATE PLOTS
        eval_plot_dir = os.path.join(eval_cfg_name_base_dir, 'plots', eval_run_cfg.start_time)
        os.makedirs(eval_plot_dir, exist_ok=True)

        plot_generator = EvalPlotGenerator(
            eval_runs_data,
            eval_plot_dir,
            img_prepend_str=eval_cfg_name,
        )
        plot_generator.plot_bonage_distribution(eval_cfg_name)
        plot_generator.plot_uncertainty_by_boneage(eval_cfg_name)
        plot_generator.plot_uncertainty_by_abs_error(eval_cfg_name)
        plot_generator.plot_abs_error_by_boneage(eval_cfg_name)
        plot_generator.plot_prediction_vs_truth(eval_cfg_name)
        plot_generator.plot_tolerated_uncertainty_abs_error(eval_cfg_name)
        plot_generator.plot_abstention_rate_vs_abs_error([eval_cfg_name])
        plot_generator.plot_reliability_de_calibration_diagram(eval_cfg_name)
        plot_generator.plot_calibration_curve([eval_cfg_name])

    logger.info('EVALUATION of DISTINCT UQ METHODS DONE!')
    logger.info('Creating combined plots...')

    # CREATE COMBINED PLOTS
    combined_plots_path = os.path.join(
        eval_run_cfg.eval_dir,
        eval_run_cfg.eval_version_name,
        'combined_plots',
        eval_run_cfg.start_time,
    )
    combined_plot_generator = EvalPlotGenerator(eval_runs_data, combined_plots_path)
    combined_plot_generator.plot_correlation_comparison(method='pearson')
    combined_plot_generator.plot_correlation_comparison(method='kendall')
    combined_plot_generator.plot_correlation_comparison(method='spearman')
    combined_plot_generator.plot_abstention_rate_vs_abs_error()
    combined_plot_generator.plot_abstention_rate_vs_abs_error(only_95_percentile=True)
    combined_plot_generator.plot_abs_error_comparison(plot_type='boxplot')
    combined_plot_generator.plot_abs_error_comparison(plot_type='violin')
    combined_plot_generator.plot_reliability_de_calibration_diagram_comparison()
    combined_plot_generator.plot_uncertainty_by_abs_error_comparison()
    combined_plot_generator.plot_abs_error_by_boneage_comparison()
    combined_plot_generator.plot_calibration_curve()

    logger.info('DONE!')


if __name__ == '__main__':
    cli_config_dict = parse_cli_args('evaluation')
    level = logging.DEBUG if cli_config_dict['debug'] else logging.INFO
    logging.basicConfig(level=level, format='%(name)s - %(asctime)s - %(levelname)s: %(message)s')

    eval_run_cfg = EvalRunConfig(cli_config_dict)
    evaluation_main(eval_run_cfg)
