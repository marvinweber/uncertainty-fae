import argparse
import logging
import pandas as pd
import os
import shutil

logger = logging.getLogger(__file__)


def main(args: argparse.Namespace) -> None:
    logger.info("START")

    annotation_dir = args.annotation_dir
    raw_image_base_dir = args.raw_image_base_dir
    target_dir = args.target_dir if args.target_dir else annotation_dir

    # Paths to directories of test and train/val images
    target_dir_train_val = os.path.join(target_dir, "train_val")
    target_dir_test = os.path.join(target_dir, "test")

    # Empty existing Target Directories and create empty ones
    for dir in [target_dir_train_val, target_dir_test]:
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir, exist_ok=True)

    df_test_annotations = pd.read_csv(
        os.path.join(annotation_dir, "test_annotations.csv"), index_col="id"
    )
    df_train_val_annotations = pd.read_csv(
        os.path.join(annotation_dir, "train_val_annotations.csv"), index_col="id"
    )
    df_train_annotations = pd.read_csv(
        os.path.join(annotation_dir, "train_annotations.csv"), index_col="id"
    )
    df_val_annotations = pd.read_csv(
        os.path.join(annotation_dir, "val_annotations.csv"), index_col="id"
    )

    # Loop Test Dataset Annotations
    for annotation_tuple in df_test_annotations.itertuples():
        img_path = copy_image(annotation_tuple, target_dir_test)
        df_test_annotations.at[annotation_tuple.Index, "img_path"] = img_path

    # Loop Train + Val Dataset Annotations
    for df in [df_train_annotations, df_val_annotations]:
        for annotation_tuple in df.itertuples():
            img_path = copy_image(annotation_tuple, target_dir_train_val, raw_image_base_dir)
            df.at[annotation_tuple.Index, "img_path"] = img_path
            df_train_val_annotations.at[annotation_tuple.Index, "img_path"] = img_path

    # Save updated Annotation CSV Files
    df_test_annotations.to_csv(os.path.join(annotation_dir, "test_annotations.csv"))
    df_train_val_annotations.to_csv(os.path.join(annotation_dir, "train_val_annotations.csv"))
    df_train_annotations.to_csv(os.path.join(annotation_dir, "train_annotations.csv"))
    df_val_annotations.to_csv(os.path.join(annotation_dir, "val_annotations.csv"))

    logger.info("DONE")


def copy_image(annotation_tuple, target_dir: str, raw_image_base_dir: str):
    image_filename = f"{annotation_tuple.Index}.png"
    source_path = os.path.join(raw_image_base_dir, image_filename)
    target_path = os.path.join(target_dir, image_filename)
    shutil.copy2(source_path, target_path)
    return target_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy images from given annotation file and update image paths (in annotation files)."
        ),
    )
    parser.add_argument(
        "annotation_dir",
        metavar="ANNOTATION_DIR",
        type=str,
        help="Directory, where the annotation files are located.",
    )
    parser.add_argument(
        "raw_image_base_dir",
        metavar="RAW_IMAGE_BASE_DIR",
        type=str,
        help=(
            "Directory, where the original (training) images of the RSNA Boneage dataset are "
            "located."
        ),
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default=None,
        required=False,
        help=(
            "Directory, where the (split) images / sets should be saved. Defaults to the "
            "annotation directory."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(name)s - %(asctime)s - %(levelname)s: %(message)s",
    )
    main(args)
