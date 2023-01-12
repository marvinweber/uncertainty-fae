import argparse
import csv
import logging
import os
import pathlib
import random
import shutil

import pandas

CSV_COLUMNS = ["id", "img_path", "boneage", "male"]

logger = logging.getLogger(__file__)


def main(args) -> None:
    logger.info("Starting...")

    save_base_dir = args.save_base_dir

    # Raw Data Paths/ Dirs
    raw_annotations_file_path = args.raw_annotations_path
    raw_image_base_dir = args.raw_image_base_dir

    # Image Directories
    train_val_dir = os.path.join(save_base_dir, "train_val")
    test_dir = os.path.join(save_base_dir, "test")

    # (Target) Annotation File Paths
    train_val_annotations_file_path = os.path.join(save_base_dir, "train_val_annotations.csv")
    train_annotations_file_path = os.path.join(save_base_dir, "train_annotations.csv")
    val_annotations_file_path = os.path.join(save_base_dir, "val_annotations.csv")
    test_annotations_file_path = os.path.join(save_base_dir, "test_annotations.csv")

    # Split Setup
    all_percentage_test = 0.1
    train_val_percentage_train = 0.8

    raw_annotations = pandas.read_csv(raw_annotations_file_path)

    if args.separate_test_images:
        logger.info("Separating Test images - Old annotations are deleted/ overwritten...")

        # Empty existing directories
        for dir in [train_val_dir, test_dir]:
            if os.path.exists(dir):
                shutil.rmtree(dir)
            os.makedirs(dir, exist_ok=True)

        # Test Annotations CSV Writer and Header Line
        test_annotations_file = open(test_annotations_file_path, "w", newline="")
        csv_writer_test_annotations = csv.writer(test_annotations_file)
        csv_writer_test_annotations.writerow(CSV_COLUMNS)

        # Train/Val Annotations CSV Writer and Header Line
        train_val_annotations_file = open(train_val_annotations_file_path, "w", newline="")
        csv_writer_train_val_annotations = csv.writer(train_val_annotations_file)
        csv_writer_train_val_annotations.writerow(CSV_COLUMNS)

        for annotation in raw_annotations.itertuples():
            filename = f"{annotation.id}.png"
            img_source_path = os.path.join(raw_image_base_dir, filename)
            img_source_path = pathlib.Path(img_source_path).as_posix()

            # Decide (randomly) whether image will be test or train/val image
            r = random.random()
            is_test_image = r <= all_percentage_test
            folder_to = test_dir if is_test_image else train_val_dir
            target_path = os.path.join(folder_to, filename)
            target_path = pathlib.Path(target_path).absolute().as_posix()

            # Copy image into test or train/val directory
            shutil.copy2(img_source_path, target_path)

            (
                csv_writer_test_annotations if is_test_image else csv_writer_train_val_annotations
            ).writerow(
                [
                    annotation.id,
                    target_path,
                    annotation.boneage,
                    annotation.male,
                ]
            )

        test_annotations_file.close()
        train_val_annotations_file.close()

    # Train Annotation Writer and Header Line
    train_annotations_file = open(train_annotations_file_path, "w", newline="")
    csv_writer_train_annotations = csv.writer(train_annotations_file)
    csv_writer_train_annotations.writerow(CSV_COLUMNS)

    # Train Annotation Writer and Header Line
    val_annotations_file = open(val_annotations_file_path, "w", newline="")
    csv_writer_val_annotations = csv.writer(val_annotations_file)
    csv_writer_val_annotations.writerow(CSV_COLUMNS)

    train_val_annotations = pandas.read_csv(train_val_annotations_file_path)
    for annotation in train_val_annotations.itertuples():
        r = random.random()
        is_train_image = r <= train_val_percentage_train

        (csv_writer_train_annotations if is_train_image else csv_writer_val_annotations).writerow(
            [
                annotation.id,
                annotation.img_path,
                annotation.boneage,
                annotation.male,
            ]
        )

    logger.info("DONE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create annotation files for RSNA Bone Age Dataset and copy/ separate the images."
        )
    )
    parser.add_argument(
        "raw_annotations_path",
        metavar="RAW_ANNOTATIONS_PATH",
        type=str,
        help=(
            "Path the the original training dataset (!) path of the RSNA Bone Age dataset. Only "
            "the training annotations are valid, as other may lack labeling."
        ),
    )
    parser.add_argument(
        "raw_image_base_dir",
        metavar="RAW_IMAGE_BASE_DIR",
        type=str,
        help="Path to the directory containing the raw original RSNA Bone Age (training) image.",
    )
    parser.add_argument(
        "save_base_dir",
        metavar="SAVE_BASE_DIR",
        type=str,
        help="Path to the (base) directory where split images and annotations should be stored.",
    )
    parser.add_argument(
        "--separate-test-images",
        required=False,
        default=True,
        action="store_true",
        help=(
            "Whether to first extract/ define given amount (percentage) of test images. This will "
            "create a fresh split and overwrite any previous split/ annotation files."
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
