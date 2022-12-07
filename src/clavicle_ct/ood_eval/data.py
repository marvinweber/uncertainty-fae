import csv
import logging
import math
import os
from typing import Optional

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import trange

from clavicle_ct.data import patient_pseudonyms_from_preprocessed_img_name
from clavicle_ct.transforms import resample_sitk_image

logger = logging.getLogger(__file__)


def generate_and_save_ood_samples(
    annotations: pd.DataFrame,
    soi_localizations: pd.DataFrame,
    image_metadata: pd.DataFrame,
    full_ct_base_dir: str,
    ood_annotation_file: str,
    ood_img_dir: str,
    patch_size: tuple[int] = (110, 60, 80),
    random_seed: Optional[int] = None,
    patches_per_ct: int = 5,
    min_required_ood_center_to_soi_distance: int = 60,
) -> None:
    """
    Generate and Store Out-of-Domain Images from Full-CT Images.

    Args:
        annotations: Annotation DataFrame of preprocessed images to used for OoD extraction.
        soi_localizations: DataFrame containing SOI centers of each Full-CT image.
        image_metadata: DataFrame containing metadata (age, sex) of each Full-CT image.
        full_ct_base_dir: Base directory of the Full-CT images/dirs.
        ood_annotation_file: Name of the annotation file to create for the OoD images.
        ood_img_dir: Directory where to store the OoD images. Must exist already.
        patch_size: Target patch size (x,y,z).
        random_seed: Optionally, a seed for the numpy random generator for deterministic patches. By
            default, no seed is used.
        patches_per_ct: Amount of patches to extract from each Full-CT image.
        min_required_ood_center_to_soi_distance: The miniumum 3D-distance required between the SOI
            center and any random patch.
    """
    max_tries_per_ct = (patches_per_ct + 1) ** 2
    patch_size_x, patch_size_y, patch_size_z = patch_size

    ood_annotation_file_s = open(ood_annotation_file, "w")
    ood_annotation_writer = csv.writer(ood_annotation_file_s)
    ood_annotation_writer.writerow(["index", "patient", "study", "series", "image", "sex", "age"])

    ood_img_index = 0
    for idx in trange(len(annotations), desc="OoD Generation Progress"):
        image_file = str(annotations.loc[idx, "image"])
        patient, study, series, full_ct_file = patient_pseudonyms_from_preprocessed_img_name(
            image_file
        )
        full_ct_file = os.path.join(full_ct_base_dir, full_ct_file)

        # Load image from disk
        try:
            image = sitk.ReadImage(full_ct_file)
            image, scaling_factor = resample_sitk_image(image)
            image = sitk.GetArrayFromImage(image)
            image = image.astype(np.float32)
        except:
            raise ValueError("Cannot load image <{:s}>".format(image_file))

        soi_x, soi_y, soi_z = _get_soi_for_pseudonyms(
            soi_localizations, patient, study, series, scaling_factor
        )
        # Full-CT Image Dimensions
        # (Direction Info is lost if SITK image is converted to numpy array -> z, y, x)
        z, y, x = image.shape[0], image.shape[1], image.shape[2]
        rng = np.random.default_rng(seed=random_seed)

        # Generate requested amount of random patches from the CT image
        ct_patches = []
        tries = 0
        for _ in range(patches_per_ct):
            patch_found = False

            while not patch_found:
                # Avoid endless loop
                if tries > max_tries_per_ct:
                    break
                tries += 1

                x0, x1 = 0, x
                y0, y1 = 0, y
                z0, z1 = 0, z

                if x > patch_size_x:
                    upper_x0_bound = x - patch_size_x
                    x0 = rng.integers(low=0, high=upper_x0_bound, endpoint=True)
                    x1 = x0 + patch_size_x
                if y > patch_size_y:
                    upper_y0_bound = y - patch_size_y
                    y0 = rng.integers(low=0, high=upper_y0_bound, endpoint=True)
                    y1 = y0 + patch_size_y
                if z > patch_size_z:
                    upper_z0_bound = z - patch_size_z
                    z0 = rng.integers(low=0, high=upper_z0_bound, endpoint=True)
                    z1 = z0 + patch_size_z

                ood_x_center = x1 - ((x1 - x0) / 2)
                ood_y_center = y1 - ((y1 - y0) / 2)
                ood_z_center = z1 - ((z1 - z0) / 2)
                ood_center_to_soi_distance = math.sqrt(
                    sum(
                        [
                            (soi_x - ood_x_center) ** 2,
                            (soi_y - ood_y_center) ** 2,
                            (soi_z - ood_z_center) ** 2,
                        ]
                    )
                )
                if ood_center_to_soi_distance >= min_required_ood_center_to_soi_distance:
                    patch_found = True

            if not patch_found:
                logger.warning(
                    "Could not found patch for (%s, %s, %s) after %s tries. Skipping Image...",
                    patient,
                    study,
                    series,
                    max_tries_per_ct,
                )
                break

            # Cropping and Sanity Check
            patch = image[z0:z1, y0:y1, x0:x1]
            assert patch.shape == tuple(reversed(patch_size)), "Generated Patch has invalid Size!"
            ct_patches.append(patch)
            logger.debug("%s Patches found after %s Tries.", patches_per_ct, tries)

        age, sex = _get_metadata_for_pseudonyms(image_metadata, patient, study, series)
        pseudonym_str = f"ae_{patient}_{study}_{series}"
        for patch in ct_patches:
            ood_img_index += 1
            filename = f"ood_{ood_img_index}_{pseudonym_str}.npy"
            patch_file_path = os.path.join(ood_img_dir, filename)
            np.save(patch_file_path, patch)
            ood_annotation_writer.writerow(
                [
                    ood_img_index,
                    patient,
                    study,
                    series,
                    filename,
                    sex,
                    age,
                ]
            )


def _get_soi_for_pseudonyms(
    soi_localizations: pd.DataFrame,
    patient: int,
    study: int,
    series: int,
    scaling_factor: np.ndarray,
) -> tuple:
    soi_row = soi_localizations.loc[(patient, study, series)]
    soi_x = soi_row["x"]
    soi_y = soi_row["y"]
    soi_z = soi_row["slice"]

    # Adjust SOI location by resampling factor
    soi_x = np.rint(soi_x * scaling_factor[0]).astype(np.int32)
    soi_y = np.rint(soi_y * scaling_factor[1]).astype(np.int32)
    soi_z = np.rint(soi_z * scaling_factor[2]).astype(np.int32)

    return soi_x, soi_y, soi_z


def _get_metadata_for_pseudonyms(
    metadata: pd.DataFrame,
    patient: int,
    study: int,
    series: int,
) -> tuple[float, str]:
    metadata_row = metadata.loc[(patient, study, series)]
    sex = metadata_row["sex"]
    age = metadata_row["age_at_acquisition"]
    return age, sex
