import csv
import logging
import os

import torch
import tqdm
from pytorch_lightning import LightningModule
from torch import Tensor

from clavicle_ct.data import undo_clavicle_age_rescale
from clavicle_ct.model_provider import ClavicleModelProvider
from rsna_boneage.data import undo_boneage_rescale
from rsna_boneage.model_provider import RSNAModelProvider
from uncertainty_fae.util import parse_cli_args
from uncertainty_fae.util.config import BaselinePredictionConfig
from uncertainty_fae.util.model_provider import ModelProvider

logger = logging.getLogger("Create-Baseline-Model-Predictions")

PROVIDER_MAPPING: dict[str, ModelProvider] = {
    "rsna_boneage": RSNAModelProvider,
    "clavicle_ct": ClavicleModelProvider,
}

UNDO_BONEAGE_SCALING_FN = {
    "rsna_boneage": undo_boneage_rescale,
    "clavicle_ct": undo_clavicle_age_rescale,
}


def main(cfg: BaselinePredictionConfig) -> None:
    logger.info("START")

    model, dm, model_provider = cfg.get_model_and_datamodule(
        PROVIDER_MAPPING,
        cfg.model_name,
        model_checkpoint=cfg.checkpoint,
    )
    assert isinstance(
        model, LightningModule
    ), "Script only works for PyTorch Lightning Models currently!"

    if cfg.dataset_type == "train":
        dm.setup("fit")
        dataloader = dm.train_dataloader()
    elif cfg.dataset_type == "val":
        dm.setup("validate")
        dataloader = dm.val_dataloader()
    elif cfg.dataset_type == "test":
        dm.setup("test")
        dataloader = dm.test_dataloader()
    else:
        raise ValueError('Invalid Dataset Type "%s"!', cfg.dataset_type)

    targets = []
    predictions = []

    model.cuda()
    model.eval()

    for x, y in tqdm.tqdm(dataloader, desc="Prediction/Batch Progress"):
        with torch.no_grad():
            pred = model.forward(x.cuda()).cpu().flatten()
            pred = UNDO_BONEAGE_SCALING_FN[cfg.get_model_data_type(cfg.model_name)](pred)

        assert isinstance(y, Tensor)
        assert isinstance(pred, Tensor)

        targets.append(y)
        predictions.append(pred)

    targets = torch.cat(targets)
    predictions = torch.cat(predictions)
    errors = torch.abs(predictions - targets)

    filepath = cfg.get_baseline_model_errors_csv_path()
    with open(filepath, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "target", "prediction", "error"])
        writer.writerows(
            zip(
                range(len(targets)),
                targets.tolist(),
                predictions.tolist(),
                errors.tolist(),
            )
        )

    logger.info("MAE: %s", float(errors.mean()))
    logger.info("DONE")


if __name__ == "__main__":
    cli_config_dict = parse_cli_args("baseline_model_predictions")
    level = logging.DEBUG if cli_config_dict["debug"] else logging.INFO
    logging.basicConfig(level=level, format="%(name)s - %(asctime)s - %(levelname)s: %(message)s")

    cfg = BaselinePredictionConfig(cli_config_dict)
    main(cfg)
