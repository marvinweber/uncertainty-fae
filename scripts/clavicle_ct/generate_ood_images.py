import argparse
import logging
import os

import pandas as pd

from clavicle_ct.ood_eval.data import generate_and_save_ood_samples

logger = logging.getLogger("Clavicle OoD Image Generation")


def main(args: argparse.Namespace) -> None:
    annotations = pd.read_csv(args.annotations)
    soi_localizations = pd.read_csv(args.soi_localizations).set_index(
        ["patient_pseudonym", "study_pseudonym", "series_pseudonym"]
    )
    image_metadata = pd.read_csv(args.image_metadata).set_index(
        ["patient_pseudonym", "study_pseudonym", "series_pseudonym"]
    )
    full_ct_base_dir = args.full_ct_base_dir
    ood_annotation_file = args.ood_annotation_file
    ood_img_dir = args.ood_img_dir
    patches_per_ct = args.patches_per_ct
    min_required_ood_center_to_soi_distance = args.min_ood_soi_distance

    logger.info("Starting OoD Generation")
    os.makedirs(ood_img_dir, exist_ok=True)
    generate_and_save_ood_samples(
        annotations=annotations,
        soi_localizations=soi_localizations,
        image_metadata=image_metadata,
        full_ct_base_dir=full_ct_base_dir,
        ood_annotation_file=ood_annotation_file,
        ood_img_dir=ood_img_dir,
        patches_per_ct=patches_per_ct,
        min_required_ood_center_to_soi_distance=min_required_ood_center_to_soi_distance,
    )
    logger.info("OoD Generation DONE!")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create Out-of-Domain Images and Annotation File from Train, Val, or Test "
            "Annotation File."
        )
    )
    parser.add_argument(
        "annotations",
        metavar="ANNOTATIONS",
        type=str,
        help=(
            "Path to the annotation file to use for OoD image generation. Files from this "
            "Annotation are used (as they should be from valid CT images) to find proper Full-CT "
            "images from which random patches are extracted."
        ),
    )
    parser.add_argument(
        "soi_localizations",
        metavar="SOI_LOCALIZATIONS",
        type=str,
        help=(
            "Path to the CSV file containing SOI (Structure of Interest) center points for each "
            "Full-CT image. This is used to ensure random patches are NOT overlapping with the "
            "SOI too much."
        ),
    )
    parser.add_argument(
        "image_metadata",
        metavar="IMAGE_METADATA",
        type=str,
        help="Path to the CSV file containing image metadata (age, sex) for each Full-CT image.",
    )
    parser.add_argument(
        "full_ct_base_dir",
        metavar="FULL_CT_BASE_DIR",
        type=str,
        help="Path to the (base) directory containing the Full-CT images/dirs.",
    )
    parser.add_argument(
        "ood_annotation_file",
        metavar="OOD_ANNOTATION_FILE",
        type=str,
        help=(
            "Path of the Annotation-CSV file to create for OoD image. Attention: any existing file "
            "will be overwritten!"
        ),
    )
    parser.add_argument(
        "ood_img_dir",
        metavar="OOD_IMG_DIR",
        type=str,
        help=(
            "Path of the directory where the OoD images should be stored. Attention: any existing "
            "images with same filename will be overwritten!"
        ),
    )
    parser.add_argument(
        "--patches-per-ct",
        type=int,
        required=False,
        default=2,
        help="Amount of Patches to extract from each CT-Image.",
    )
    parser.add_argument(
        "--min-ood-soi-distance",
        type=int,
        required=False,
        default=60,
        help="Minimum required distance of random patches (center) to SOI center.",
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
