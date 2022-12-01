import csv
import logging
import os
import re
from typing import Callable, Optional, TypedDict

import torch
import yaml
from pandas import DataFrame, Series
from torch import Tensor
from torch.utils.data import DataLoader

from uncertainty_fae.model import (ADT_STAT_MEAN_UNCERTAINTY, ADT_STAT_PREDS_DISTINCT,
                                   UncertaintyAwareModel)

logger = logging.getLogger(__name__)

EVAL_PREDICTION_COLUMNS = ['target', 'prediction', 'error', 'uncertainty']
EVAL_DISTINCT_PREDICTIONS_COLUMNS = ['index', 'target', 'prediction', 'error']


def generate_evaluation_predictions(
    eval_base_dir: str,
    eval_model: UncertaintyAwareModel,
    dataloader: DataLoader,
) -> tuple[str, str, str]:
    """
    TODO: Documentation

    Returns:
        A tuple (eval_results_file, eval_predictions_file, eval_single_predictions_file) where
        each one is either a file path to the corresponding file or None, if this type of file/
        result could not be provided (e.g., if no single predictions are available, this file
        will not exist).
    """
    exists, eval_result_file, eval_predictions_file, eval_distinct_predictions_file = \
        evaluation_predictions_available(eval_base_dir, make_eval_dir=True)

    if exists:
        return eval_result_file, eval_predictions_file, eval_distinct_predictions_file

    results = eval_model.evaluate_dataset(dataloader)
    score, predictions, targets, errors, uncertainties, metrics = results

    # Result Stats
    eval_result_stats = {'score': float(score)}
    if ADT_STAT_MEAN_UNCERTAINTY in metrics:
        eval_result_stats[ADT_STAT_MEAN_UNCERTAINTY] = float(metrics[ADT_STAT_MEAN_UNCERTAINTY])

    # Distinct Predictions, if available...
    if ADT_STAT_PREDS_DISTINCT in metrics:
        with open(eval_distinct_predictions_file, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(EVAL_DISTINCT_PREDICTIONS_COLUMNS)

            for i, img_preds in enumerate(metrics[ADT_STAT_PREDS_DISTINCT]):
                assert isinstance(img_preds, Tensor)
                tensor_size = (len(img_preds), )
                img_target = torch.full(tensor_size, targets[i])
                img_errors = torch.abs(img_preds - img_target)
                index_list = [i for _ in range(len(img_preds))]
                writer.writerows(
                    zip(index_list, img_target.tolist(), img_preds.tolist(), img_errors.tolist())
                )
    # ...otherwise, return None for the corresponding file path.
    else:
        eval_distinct_predictions_file = None

    # Prediction and Uncertainty Stats
    with open(eval_predictions_file, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(EVAL_PREDICTION_COLUMNS)
        writer.writerows(
            zip(targets.tolist(), predictions.tolist(), errors.tolist(), uncertainties.tolist())
        )

    # Eval Result Yaml File
    with open(eval_result_file, 'w') as file:
        yaml.dump(eval_result_stats, file)

    return eval_result_file, eval_predictions_file, eval_distinct_predictions_file


def evaluation_predictions_available(eval_base_dir: str, make_eval_dir: bool = False) -> bool:
    eval_dir = os.path.join(eval_base_dir, 'predictions')
    eval_result_file = os.path.join(eval_dir, 'eval_result.yml')
    eval_predictions_file = os.path.join(eval_dir, 'eval_predictions.csv')
    eval_distinct_predictions_file = os.path.join(eval_dir, 'eval_distinct_predictions.csv')

    if make_eval_dir:
        os.makedirs(eval_dir, exist_ok=True)

    if os.path.exists(eval_result_file):
        eval_distinct_predictions_file = (eval_distinct_predictions_file
                                          if os.path.exists(eval_distinct_predictions_file)
                                          else None)
        return True, eval_result_file, eval_predictions_file, eval_distinct_predictions_file
    return False, eval_result_file, eval_predictions_file, eval_distinct_predictions_file


def create_best_epoch_checkpoint_symlinks(base_dir: str, symlink_name: str = 'best.ckpt') -> None:
    """Create symlinks to best epoch checkpoints in every checkpoint dir found under base_dir.

    Note, this only works if the loss is written to the file itself: `val_loss=...`;
    e.g. `epoch=8-val_loss=0.123.ckpt`.

    Args:
        base_dir: The directory in which to (recursively) search for `checkpoints` directories in
            which to add the symlinks.
        symlink_name: How the symlink should be named.
    """
    for dirname, dir_dirs, dir_files in os.walk(base_dir):
        if not dirname.endswith('checkpoints'):
            continue
        checkpoint_files = []
        for dir_file in dir_files:
            result = re.search(r'val_loss=(-?\d*\.\d*)', dir_file)
            if not result:
                continue
            val_loss = float(result.group(1))
            checkpoint_files.append((val_loss, dir_file))
        if not checkpoint_files:  # No checkpoint files found
            continue
        best_ckpt_path = os.path.abspath(
            os.path.join(
                dirname,
                sorted(checkpoint_files, key=lambda cpt: cpt[0])[0][1],
        ))
        symlink_path = os.path.abspath(os.path.join(dirname, symlink_name))
        # Make sure to not overwrite any file/dir
        if not os.path.isfile(symlink_path) and not os.path.isdir(symlink_path):
            # remove old link to ensure it is up to date
            if os.path.islink(symlink_path):
                os.remove(symlink_path)
            os.symlink(best_ckpt_path, symlink_path)


def apply_df_age_transform(
    df: DataFrame,
    transform_fn: Callable[[Series], Series],
    columns: Optional[list[str]] = None,
) -> DataFrame:
    """
    Apply (Age) Tranformation to Evaluation DataFrame (e.g., convert days to years).

    Can be used for any kind of transformation, however, default columns used correspond to age
    transformation.

    Args:
        df: The dataframe to apply the transform to.
        transform_fn: A function that transforms a single series (i.e., a single pandas column).
        columns: A list of column names to tranform. If not given, default evaluation columns will
            be used. Transform is skipped for column names not available in `df`.

    Returns:
        A DataFrame where the transform has been applied to all age related columns.
    """
    if not columns:
        columns = EVAL_PREDICTION_COLUMNS

    for col in columns:
        if col not in df:
            continue
        df[col] = transform_fn(df[col])
    return df


class EvalRunData(TypedDict):
    """Simple Wrapper for results data as required by the `EvalPlotGenerator`."""

    display_name: str
    data_display_name: Optional[str]
    prediction_log: DataFrame
    distinct_prediction_log: Optional[DataFrame]
    color: str
