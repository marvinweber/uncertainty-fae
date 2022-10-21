import argparse
import csv
import os
import pathlib
import re
from datetime import datetime

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from util.eval import print_progress
from util.eval_plot_generator import EvalPlotGenerator
from util.model_provider import get_model_and_datamodule

parser = argparse.ArgumentParser(description='Training of RSNA Boneage Models')
parser.add_argument('train_log', metavar='TRAIN_LOG', type=str,
                    help='Log of the model training (csv file).')
parser.add_argument('--mc-iterations', metavar='MC_ITERATIONS', type=int, default=150,
                    required=False,
                    help='Amount of predictions to use for MC Averaging (MC Dropout).')

USE_CUDA = True
LOG_COLUMNS = ['image_id', 'boneage', 'predicted_boneage', 'abs_error', 'uncertainty',
               'uncertainty_17_83', 'uncertainty_5_95', 'predictions_median']


def evaluate_all(train_log_filepath: str, mc_iterations: int):
    save_dir = 'eval_logs'
    eval_version = datetime.now().strftime('%Y-%m-%d-%H-%M')
    train_version_ts = re.findall('\d\d\d\d\-\d\d\-\d\d\-\d\d\-\d\d',
                                  os.path.basename(train_log_filepath))[0]
    train_version = f'{train_version_ts}_mc{mc_iterations}'
    train_log = pd.read_csv(train_log_filepath, index_col='model_id')
    eval_log = train_log.copy(True).drop(columns=['max_epochs', 'early_stopping_patience',
                                                  'log_dir','checkpoint_dir'])
    eval_log_file = os.path.join(save_dir, f'eval_log_{train_version}_{eval_version}.csv')

    print('Found Train-Version:', train_version_ts)
    print('Will log summary to:', eval_log_file)

    for train_log_entry in train_log.itertuples():
        try:
            print('\n######### NEXT MODEL ###############')
            if not train_log_entry.log_name or not isinstance(train_log_entry.log_name, str):
                print('Skip Model:', train_log_entry.log_name, '- No Log Dir found!')
                continue
            print('Model:', train_log_entry.log_name)

            metrics_df = pd.read_csv(os.path.join(train_log_entry.log_dir, 'metrics.csv'))
            metrics_df = metrics_df.dropna(subset=['val_loss'])
            best_epoch = next(metrics_df.sort_values('val_loss', ascending=True).head(1).itertuples())
            last_epoch = next(metrics_df.sort_values('epoch', ascending=False).head(1).itertuples())
            best_epoch_checkpoint = os.path.join(
                train_log_entry.checkpoint_dir,
                _get_checkpoint_name(best_epoch.epoch, best_epoch.val_loss))
            last_epoch_checkpoint = os.path.join(
                train_log_entry.checkpoint_dir,
                _get_checkpoint_name(last_epoch.epoch, last_epoch.val_loss))

            eval_base_dir = os.path.join(save_dir, train_log_entry.log_name, train_version)
            best_epoch_dir = os.path.join(eval_base_dir, pathlib.Path(best_epoch_checkpoint).stem)
            last_epoch_dir = os.path.join(eval_base_dir, pathlib.Path(last_epoch_checkpoint).stem)

            for dir in [save_dir, best_epoch_dir, last_epoch_dir]:
                os.makedirs(dir, exist_ok=True)

            best_epoch_pred_file = os.path.join(best_epoch_dir, f'predictions.csv')
            _generate_model_predictions(
                train_log_entry, best_epoch_pred_file, best_epoch_checkpoint, mc_iterations)
            best_epoch_eval_dir = os.path.join(best_epoch_dir, f'eval_{eval_version}')
            _eval_single_model(
                best_epoch_pred_file, best_epoch_eval_dir, train_log_entry.log_name,
                eval_log_df=eval_log, eval_log_df_index=train_log_entry.Index)

            # Only evaluate last epoch, if it was not the best epoch
            if best_epoch.epoch != last_epoch.epoch and False:
                # TODO: fix issues with index (-> eval log file / df) / use second log file
                last_epoch_pred_file = os.path.join(last_epoch_dir, 'predictions.csv')
                _generate_model_predictions(
                    train_log_entry, last_epoch_pred_file, last_epoch_checkpoint, mc_iterations)
                _eval_single_model()
        except Exception as e:
            print('!!!!!!!!!!!!!! EXCEPTION !!!!!!!!!!!!!!')
            print(e)
        else:
            print('Evaluation of model finished!')

        eval_log.to_csv(eval_log_file)


def _get_checkpoint_name(epoch: int, val_loss: float):
    return f'epoch={epoch}-val_loss={val_loss:2f}.ckpt'


def _generate_model_predictions(
        train_log_entry, model_predictions_log_file: str, checkpoint_file: str, mc_iterations: int):
    if os.path.exists(model_predictions_log_file):
        print('Prediction Log already exists:', train_log_entry.name)
        return

    print('Generating predictions for:', train_log_entry.name)
    img_input_dimensions = (train_log_entry.img_input_width, train_log_entry.img_input_height)
    model, dm = get_model_and_datamodule(
        train_log_entry.name,
        img_input_dimensions,
        train_log_entry.with_gender_input,
        rescale_boneage=False,
        rebalance_classes=False,
        checkpoint_path=checkpoint_file,
        litmodel_kwargs={'mc_iterations': mc_iterations, 'undo_boneage_rescaling': True})

    model.train(False)
    dm.setup('validate')
    data = dm.dataset_val
    test_dataloader = DataLoader(data, batch_size=1, shuffle=False)

    log_file_h = open(model_predictions_log_file, 'w', newline='')
    log_file_csv_writer = csv.writer(log_file_h)
    log_file_csv_writer.writerow(LOG_COLUMNS)

    if USE_CUDA:
        model.cuda()

    start = datetime.now()

    for batch, (X, y) in enumerate(test_dataloader):
        if USE_CUDA:
            if isinstance(X, list):
                X = [xs.cuda() for xs in X]
            else:
                X = X.cuda()

        prediction, metrics = model.forward_with_mc(X)
        # X.cpu()

        boneage = y.detach().numpy()[0]
        prediction = prediction.cpu().detach().numpy()

        abs_error = abs(boneage - prediction)
        uncertainty = metrics['uncertainty'].detach().numpy()
        log_file_csv_writer.writerow([
            'unkown',
            boneage,
            prediction,
            abs_error,
            uncertainty,
            metrics['uncertainty_17_83_range'].detach().numpy(),
            metrics['uncertainty_5_95_range'].detach().numpy(),
            metrics['median'].detach().numpy(),
        ])

        if (batch + 1) % 50 == 0:
            print_progress(batch+1, len(test_dataloader), start)

    log_file_h.close()


def _eval_single_model(model_predictions_log_file: str, eval_results_dir: str, name: str,
                       eval_log_df: pd.DataFrame, eval_log_df_index: pd.Index):
    log_df = pd.read_csv(model_predictions_log_file)
    plot_generator = EvalPlotGenerator(log_df, eval_results_dir, img_prepend_str=name)
    os.makedirs(eval_results_dir, exist_ok=True)

    print('++++++++ STATS ++++++++')
    print(f'MIN PREDICTION:   {log_df["predicted_boneage"].min()}')
    print(f'MAX PREDICTION:   {log_df["predicted_boneage"].max()}')
    print(f'MIN ABS ERROR:    {log_df["abs_error"].min()}')
    print(f'MAX ABS ERROR:    {log_df["abs_error"].max()}')
    print(f'MEAN ABS ERROR:   {log_df["abs_error"].mean()}')
    print(f'MEDIAN ABS ERROR: {log_df["abs_error"].median()}')
    print(f'MIN UNCERTAINTY:  {log_df["uncertainty"].min()}')
    print(f'MAX UNCERTAINTY:  {log_df["uncertainty"].max()}')
    print('++++++++ STATS ++++++++')
    eval_log_df.at[eval_log_df_index, 'prediction_min'] = log_df['predicted_boneage'].min()
    eval_log_df.at[eval_log_df_index, 'prediction_max'] = log_df['predicted_boneage'].max()
    eval_log_df.at[eval_log_df_index, 'abs_error_min'] = log_df['abs_error'].min()
    eval_log_df.at[eval_log_df_index, 'abs_error_max'] = log_df['abs_error'].max()
    eval_log_df.at[eval_log_df_index, 'abs_error_mean'] = log_df['abs_error'].mean()
    eval_log_df.at[eval_log_df_index, 'abs_error_median'] = log_df['abs_error'].median()
    eval_log_df.at[eval_log_df_index, 'uncertainty_min'] = log_df['uncertainty'].min()
    eval_log_df.at[eval_log_df_index, 'uncertainty_max'] = log_df['uncertainty'].max()
    eval_log_df.at[eval_log_df_index, 'uncertainty_mean'] = log_df['uncertainty'].mean()
    eval_log_df.at[eval_log_df_index, 'uncertainty_median'] = log_df['uncertainty'].median()

    plot_generator.plot_abs_error_uncertainty_scatter()
    plot_generator.plot_abs_error_uncertainty_17_83_scatter()
    plot_generator.plot_bonage_distribution()
    plot_generator.plot_uncertainty_by_boneage()
    plot_generator.plot_uncertainty_comparison()
    plot_generator.plot_abs_error_by_boneage()
    plot_generator.plot_uncertainty_by_abs_error()
    plot_generator.plot_prediction_vs_truth()
    plot_generator.plot_tolerated_uncertainty_abs_error()


if __name__ == '__main__':
    args = parser.parse_args()
    train_log_filepath = args.train_log
    mc_iterations = args.mc_iterations
    evaluate_all(train_log_filepath, mc_iterations)
