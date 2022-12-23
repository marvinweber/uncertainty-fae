import logging
import os

import pandas as pd
import yaml

from clavicle_ct.model_provider import ClavicleModelProvider
from clavicle_ct.ood_eval import ClavicleCtOutOfDomainEvaluator
from rsna_boneage.model_provider import RSNAModelProvider
from rsna_boneage.ood_eval import RSNABoneAgeOutOfDomainEvaluator
from uncertainty_fae.evaluation import EvalPlotGenerator, OutOfDomainEvaluator
from uncertainty_fae.evaluation.plotting import EvalRunData
from uncertainty_fae.evaluation.util import (
    apply_df_age_transform,
    create_best_epoch_checkpoint_symlinks,
    evaluation_predictions_available,
    generate_evaluation_predictions,
)
from uncertainty_fae.model import UncertaintyAwareModel
from uncertainty_fae.util import EvalRunConfig, parse_cli_args
from uncertainty_fae.util.model_provider import ModelProvider

logger = logging.getLogger('UNCERTAINTY_FAE_EVALUATION')

PROVIDER_MAPPING: dict[str, ModelProvider] = {
    'rsna_boneage': RSNAModelProvider,
    'clavicle_ct': ClavicleModelProvider,
}

OOD_EVALUATOR_MAPPING: dict[str, OutOfDomainEvaluator] = {
    'rsna_boneage': RSNABoneAgeOutOfDomainEvaluator,
    'clavicle_ct': ClavicleCtOutOfDomainEvaluator,
}

DATASET_NAMES = {
    'rsna_boneage': 'RSNA Bone Age',
    'clavicle_ct': 'Clavicle CT',
}

DATASET_AGE_TO_YEAR_TRANSFORMS = {
    'rsna_boneage': lambda df_ser: df_ser / 12,  # month -> year
    'clavicle_ct': lambda df_ser: df_ser / 365.25,  # days -> year
}

DATASET_AGE_TO_YEAR_TRANSFORMS_UNDO = {
    'rsna_boneage': lambda df_ser: df_ser * 12,  # year -> month
    'clavicle_ct': lambda df_ser: df_ser * 365.25,  # year -> days
}

def evaluation_main(eval_run_cfg: EvalRunConfig) -> None:
    if len(eval_run_cfg.eval_only) == 0:
        logger.error('No Evaluation Configs could be found!')
        return
    else:
        logger.info('Running Evaluations for: %s', ', '.join(eval_run_cfg.eval_only))

    if eval_run_cfg.model_logs_dir:
        logger.info('Creating best-epoch symlinks for all checkpoint dirs found in %s',
                    eval_run_cfg.model_logs_dir)
        create_best_epoch_checkpoint_symlinks(eval_run_cfg.model_logs_dir)

    eval_runs_data: dict[str, EvalRunData] = {}

    eval_base_dir = os.path.join(  # Base directory for overall evaluation
        eval_run_cfg.eval_dir,
        eval_run_cfg.eval_version_name,
        eval_run_cfg.dataset_type,
    )

    # Determine Data-Type (RSNA Bone Age, Clavicle CT, ...) for this Evaluation Run
    # We assume same data type for entire evaluation (no plots / evals accross different datasets)
    first_eval_model = list(eval_run_cfg.eval_configuration.values())[0]['model']
    data_type = eval_run_cfg.model_configurations[first_eval_model]['data']

    # Base directory and Evaluator for out of domain tests
    # OOD Data must not be stored per data_type, as the OOD-datasets stay always the same.
    ood_base_dir_data = os.path.join(eval_run_cfg.eval_dir, eval_run_cfg.eval_version_name, 'ood')
    ood_base_dir_plots = os.path.join(eval_base_dir, 'ood_eval')
    ood_evaluator_cls = OOD_EVALUATOR_MAPPING[data_type]
    ood_evaluator = ood_evaluator_cls.get_evaluator(
        ood_base_dir_data,
        ood_base_dir_plots,
        eval_run_cfg,
        DATASET_AGE_TO_YEAR_TRANSFORMS[data_type],
    )

    for eval_cfg_name, eval_cfg in eval_run_cfg.eval_configuration.items():
        if eval_cfg_name not in eval_run_cfg.eval_only:
            logger.debug('Skipping config %s', eval_cfg_name)
            continue
        logger.info('NEXT EVALUATION --- %s (%s) for model "%s"',
                    eval_cfg_name, eval_cfg['name'], eval_cfg['model'])

        eval_cfg_name_base_dir = os.path.join(eval_base_dir, eval_cfg_name)
        logger.info('Eval Log-Dir: %s', eval_cfg_name_base_dir)

        # GENERATING PREDICTIONS
        avail, *eval_files = evaluation_predictions_available(eval_cfg_name_base_dir)
        ood_avail = ood_evaluator.ood_preds_avail(eval_cfg_name)
        if not avail or not ood_avail:
            # Skip, if no predictions should be made
            if eval_run_cfg.only_plotting:
                logger.warning(
                    'Only Plotting (--only-plotting) enabled, but %s is missing predictions! Skip.',
                    eval_cfg_name,
                )
                continue

            model, dm, model_provider = eval_run_cfg.get_model_and_datamodule(
                PROVIDER_MAPPING,
                eval_cfg['model'],
                model_checkpoint=eval_cfg['checkpoint'],
                eval_cfg_name=eval_cfg_name,
            )
            assert isinstance(model, UncertaintyAwareModel)
            dm.setup('all')
            model.set_dataloaders(
                train_dataloader=dm.train_dataloader(),
                val_dataloader=dm.val_dataloader(),
            )

            # Start with OoD Predictions
            if not ood_avail:
                logger.info('Generating OoD Predictions...')
                ood_evaluator.generate_predictions(eval_cfg_name, model, model_provider)
                logger.info('OoD Predictions GENERATED.')
            else:
                logger.info('OoD Predictions already available; skipping.')

            # Continue with "normal" predictions
            if not avail:
                logger.info('Generating Evaluation Predictions...')
                dataloader = eval_run_cfg.get_eval_dataloader(dm)
                eval_files = generate_evaluation_predictions(
                    eval_cfg_name_base_dir,
                    model,
                    dataloader,
                )
                logger.info('Evaluation Predictions GENERATED.')
            else:
                logger.info('Evaluation Predictions already available; skipping.')
        else:
            logger.info('SKIPPING PREDICTIONS, as all are already available!')
        eval_result_file, eval_predictions_file, eval_distinct_predictions_file = eval_files

        prediction_log = apply_df_age_transform(
            pd.read_csv(eval_predictions_file),
            DATASET_AGE_TO_YEAR_TRANSFORMS[data_type],
        )
        distinct_prediction_log = None
        if eval_distinct_predictions_file:
            distinct_prediction_log = apply_df_age_transform(
                pd.read_csv(eval_distinct_predictions_file),
                DATASET_AGE_TO_YEAR_TRANSFORMS[data_type],
            )
        eval_runs_data[eval_cfg_name] = {
            'display_name': eval_cfg['name'],
            'data_display_name': DATASET_NAMES[data_type],
            'prediction_log': prediction_log,
            'distinct_prediction_log': distinct_prediction_log,
            'color': eval_cfg['color'] if 'color' in eval_cfg else 'black',
            'marker': eval_cfg['marker'] if 'marker' in eval_cfg else 'D',
        }

        if eval_run_cfg.only_combined_plots or eval_run_cfg.only_predictions:
            continue

        logger.info('Evaluating Single Model...')

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
            undo_age_to_year_transform=DATASET_AGE_TO_YEAR_TRANSFORMS_UNDO[data_type]
        )
        plot_generator.plot_age_distribution(eval_cfg_name)
        plot_generator.plot_uncertainty_by_age(eval_cfg_name)
        plot_generator.plot_uncertainty_by_error(eval_cfg_name)
        plot_generator.plot_uncertainty_by_error(eval_cfg_name, with_aucs=True)
        plot_generator.plot_error_by_age(eval_cfg_name)
        plot_generator.plot_prediction_vs_truth(eval_cfg_name)
        plot_generator.plot_error_by_abstention_rate([eval_cfg_name])
        plot_generator.plot_calibration_curve([eval_cfg_name])

    logger.info('DISTINCT UQ METHODS/MODELS DONE!')

    if eval_run_cfg.only_predictions:
        logger.info('All Predictions done; DONE!')
        return

    # CREATE COMBINED PLOTS
    logger.info('Creating combined plots...')
    combined_plots_path = os.path.join(
        eval_base_dir,
        'combined_plots',
        eval_run_cfg.start_time,
    )
    baseline_model_error_df = eval_run_cfg.get_baseline_error_dataframe(data_type)
    if baseline_model_error_df is not None:
        baseline_model_error_df = apply_df_age_transform(
            baseline_model_error_df,
            DATASET_AGE_TO_YEAR_TRANSFORMS[data_type],
            ['error'],
        )
    mean_predictor_error_df = eval_run_cfg.get_mean_predictor_error_dataframe(data_type)
    if mean_predictor_error_df is not None:
        mean_predictor_error_df = apply_df_age_transform(
            mean_predictor_error_df,
            DATASET_AGE_TO_YEAR_TRANSFORMS[data_type],
            ['error'],
        )
    combined_plot_generator = EvalPlotGenerator(
        eval_runs_data,
        combined_plots_path,
        undo_age_to_year_transform=DATASET_AGE_TO_YEAR_TRANSFORMS_UNDO[data_type],
        baseline_model_error_df=baseline_model_error_df,
        mean_predictor_model_error_df=mean_predictor_error_df,
        img_prepend_str=data_type,
    )
    combined_plot_generator.plot_correlation_comparison(method='pearson')
    combined_plot_generator.plot_correlation_comparison(method='kendall')
    combined_plot_generator.plot_correlation_comparison(method='spearman')
    combined_plot_generator.plot_error_by_abstention_rate()
    combined_plot_generator.plot_error_by_abstention_rate(only_p95=True)
    combined_plot_generator.plot_error_comparison(plot_type='boxplot')
    combined_plot_generator.plot_error_comparison(plot_type='violin')
    combined_plot_generator.plot_uncertainty_by_error_comparison()
    combined_plot_generator.plot_uncertainty_by_error_aucs_comparison(plot_type="mean")
    combined_plot_generator.plot_uncertainty_by_error_aucs_comparison(plot_type="min")
    combined_plot_generator.plot_uncertainty_by_error_aucs_comparison(plot_type="mean_min")
    combined_plot_generator.save_uncertainty_by_error_aucs_csv()
    combined_plot_generator.save_uncertainty_reorder_ranks_csv()
    combined_plot_generator.plot_error_by_age_comparison()
    combined_plot_generator.plot_calibration_curve(comparison_plot=True)
    combined_plot_generator.save_error_uncertainty_stats()

    logger.info('Creating OOD Plots...')
    ood_evaluator.generate_plots(eval_runs_data)

    logger.info('DONE!')


if __name__ == '__main__':
    cli_config_dict = parse_cli_args('evaluation')
    level = logging.DEBUG if cli_config_dict['debug'] else logging.INFO
    logging.basicConfig(level=level, format='%(name)s - %(asctime)s - %(levelname)s: %(message)s')

    eval_run_cfg = EvalRunConfig(cli_config_dict)
    evaluation_main(eval_run_cfg)
