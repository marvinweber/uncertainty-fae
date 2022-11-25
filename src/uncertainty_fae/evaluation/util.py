import csv
import logging
import os
from typing import Optional, TypedDict

import torch
import yaml
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import DataLoader

from uncertainty_fae.model import (ADT_STAT_MEAN_UNCERTAINTY, ADT_STAT_PREDS_DISTINCT,
                                   UncertaintyAwareModel)

logger = logging.getLogger(__name__)


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
            writer.writerow(['index', 'target', 'prediction', 'error'])

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
        writer.writerow(['target', 'prediction', 'error', 'uncertainty'])
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


class EvalRunData(TypedDict):
    """Simple Wrapper for results data as required by the `EvalPlotGenerator`."""

    display_name: str
    data_display_name: Optional[str]
    prediction_log: DataFrame
    distinct_prediction_log: Optional[DataFrame]
    color: str
