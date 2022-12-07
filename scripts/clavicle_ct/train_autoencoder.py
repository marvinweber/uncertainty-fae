import argparse
import logging
import os
from datetime import datetime

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from clavicle_ct.data import CTDataModule
from clavicle_ct.litmodel.autoencoder import LitClavicleAutoencoder
from clavicle_ct.net.autoencoder import (
    agenet14_3d_autoencoder,
    agenet18_3d_autoencoder,
    agenet18_3d_light_autoencoder,
    agenet34_3d_autoencoder,
)
from clavicle_ct.transforms import get_autoencoder_transforms

logger = logging.getLogger("AgeNet-Autoencoder-Training")


def main(args: argparse.Namespace) -> None:
    version = args.version
    annotation_file_train = args.annotation_file_train
    annotation_file_val = args.annotation_file_val
    preprocessed_img_base_dir = args.preprocessed_img_base_dir
    full_ct_base_dir = args.full_ct_base_dir

    max_epochs = args.max_epochs
    early_stopping_patience = args.early_stopping_patience
    batch_size = args.batch_size
    dataloading_num_workers = args.dataloader_num_workers

    save_base_dir = args.save_dir

    # Currently hardcoded, as it is the only tested network architecture.
    network_name = "agenet18_3d_autoencoder"

    # Currently hardcoded: The autoencoder learns to encode/decode patches independently from the
    # sex (only tested approach).
    sex_input = False

    start_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")

    # Path Setup
    save_dir = os.path.join(save_base_dir, network_name, str(version))
    checkpoint_dir = os.path.join(save_dir, "checkpoints")
    last_checkpoint_path = os.path.join(checkpoint_dir, "last.ckpt")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Check if Training is resumed
    is_resume = False
    if os.path.exists(last_checkpoint_path):
        is_resume = True

    # Logger
    tb_logger = TensorBoardLogger(save_dir=save_dir, name="tensorboard", version=start_time)
    csv_logger = CSVLogger(save_dir=save_dir, name="metrics", version=start_time)

    # Callbacks
    model_save_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor="val_loss",
        save_last=True,
        save_top_k=2,
        filename="{step}-{epoch}-{val_loss:.2f}",
        save_on_train_epoch_end=True,
    )
    early_stopping_cb = EarlyStopping(
        monitor="val_loss",
        patience=early_stopping_patience,
        mode="min",
    )

    # Datamodule Setup
    ct_dm = CTDataModule(
        annotation_file_train=annotation_file_train,
        annotation_file_val=annotation_file_val,
        img_train_base_dir=preprocessed_img_base_dir,
        img_val_base_dir=preprocessed_img_base_dir,
        full_ct_image_base_dir=full_ct_base_dir,
        use_full_cts=True,
        batch_size=batch_size,
        include_sex=sex_input,
        transforms_input=get_autoencoder_transforms("input"),
        transforms_target=get_autoencoder_transforms("target"),
        num_workers=dataloading_num_workers,
    )
    ct_dm.prepare_data()
    ct_dm.setup(stage="fit")

    # Network
    if network_name == "agenet14_3d_autoencoder":
        network = agenet14_3d_autoencoder(use_sex=sex_input)
    elif network_name == "agenet18_3d_autoencoder":
        network = agenet18_3d_autoencoder(use_sex=sex_input)
    elif network_name == "agenet18_3d_light_autoencoder":
        network = agenet18_3d_light_autoencoder(use_sex=sex_input)
    elif network_name == "agenet34_3d_autoencoder":
        network = agenet34_3d_autoencoder(use_sex=sex_input)
    else:
        raise ValueError("Network name <{:s}> not accepted.".format(network_name))
    n_inputs = 1 if not network.use_sex == True else 2

    model = LitClavicleAutoencoder(net=network, n_inputs=n_inputs)
    trainer = Trainer(
        logger=[csv_logger, tb_logger],
        callbacks=[early_stopping_cb, model_save_cb],
        accelerator="gpu",
        devices=1,
        max_epochs=max_epochs,
        reload_dataloaders_every_n_epochs=1,  # Ensure to resample the CT Datasets
    )

    if not is_resume:
        logger.info("Start new Training...")
        trainer.fit(model, ct_dm)
    else:
        logger.info("RESUME Training...")
        trainer.fit(model, ct_dm, ckpt_path=last_checkpoint_path)

    logger.debug("Trainer Fit returned")
    if trainer.interrupted:
        logger.warning("Training was interrupted! Maybe continue is possible.")
    else:
        logger.info("Training FINISHED sucessfully!")

    logger.info("Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the AgeNet Autoencoder.")
    parser.add_argument("version", metavar="VERSION", type=int, help="Version Name.")
    parser.add_argument(
        "--annotation-file-train",
        type=str,
        required=False,
        default="/data_fae_uq/clavicle_ct/annotations_train.csv",
        help="Path to the Train-Annotations of the (preprocessed) Clavicle CT dataset.",
    )
    parser.add_argument(
        "--annotation-file-val",
        type=str,
        required=False,
        default="/data_fae_uq/clavicle_ct/annotations_val.csv",
        help="Path to the Validation-Annotations of the (preprocessed) Clavicle CT dataset.",
    )
    parser.add_argument(
        "--preprocessed-img-base-dir",
        type=str,
        required=False,
        default="/data_fae_uq/clavicle_ct/preprocessed",
        help="Path to the directory containing the preprocessed images of the Clavicle CT dataset.",
    )
    parser.add_argument(
        "--full-ct-base-dir",
        type=str,
        required=False,
        default="/data_fae_uq/clavicle_ct/images",
        help="Path to the directory containing the raw full CT images of the Clavicle CT dataset.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="/ml_logs",
        required=False,
        help="Directory to save training logs (checkpoints, metrics) to.",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=500, required=False, help="Maximum Epochs to train."
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        required=False,
        help="Patience for EarlyStopping Callback.",
    )
    parser.add_argument("--batch-size", type=int, required=False, default=8)
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        required=False,
        default=12,
        help="Amount of workers to use for dataloading.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        required=False,
        default=False,
        help="Flag to enable output of DEBUG logs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format="%(name)s - %(asctime)s - %(levelname)s: %(message)s")
    main(args)
