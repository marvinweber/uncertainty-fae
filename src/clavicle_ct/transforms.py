import logging

import numpy as np
import SimpleITK as sitk
import torchio as tio

logger = logging.getLogger(__file__)


def get_transforms(
    augmentation: bool = False,
    level: int = 1,
    input_resize: tuple = (150, 150, 150),
    out_shape: tuple = (112, 112, 112),
    ct_window: tuple = (500, 1500),
) -> tio.Compose:

    lower_hu_bound = ct_window[0] - int(ct_window[1] / 2.0)
    upper_hu_bound = ct_window[0] + int(ct_window[1] / 2.0)

    # Preprocessing transforms before augmentation
    pre_aug_preprocessing_tansforms = tio.Compose(
        [
            tio.transforms.Resize(input_resize),
            tio.transforms.Clamp(out_min=lower_hu_bound, out_max=upper_hu_bound),
            tio.transforms.RescaleIntensity(
                out_min_max=(-1, 1), in_min_max=(lower_hu_bound, upper_hu_bound)
            ),
        ]
    )

    # Augmentation transforms
    if level == 1:
        augmentation_tansforms = tio.Compose(
            [
                tio.transforms.RandomFlip(axes=2),  # Only flip left and right
                tio.transforms.RandomAffine(
                    scales=(0.05, 0.05, 0.05),  # factor
                    degrees=(5, 5, 5),  # degree
                    translation=(5, 5, 5),  # voxel
                    center="image",
                    default_pad_value="minimum",
                ),
                tio.transforms.RandomNoise(mean=0.0, std=0.01),
            ]
        )
    elif level == 2:
        augmentation_tansforms = tio.Compose(
            [
                tio.transforms.RandomFlip(axes=(0, 1, 2)),  # Only flip left and right
                tio.transforms.RandomAffine(
                    scales=(0.1, 0.1, 0.1),  # factor
                    degrees=(10, 10, 10),  # degree
                    translation=(10, 10, 10),  # voxel
                    center="image",
                    default_pad_value="minimum",
                ),
                tio.transforms.RandomNoise(mean=0.0, std=0.02),
            ]
        )
    else:
        raise ValueError("Augmentation level <{:d}> is not supported.".format(level))

    # Preprocessing transforms after augmentating
    post_aug_preprocessing_tansforms = tio.Compose(
        [tio.transforms.CropOrPad(target_shape=out_shape)]
    )

    # Preprocessing without augmentation (default)
    if not augmentation:
        transforms = tio.Compose(
            [pre_aug_preprocessing_tansforms, post_aug_preprocessing_tansforms]
        )
        return transforms
    # Preprocessing with augmentation
    else:
        transforms = tio.Compose(
            [
                pre_aug_preprocessing_tansforms,
                augmentation_tansforms,
                post_aug_preprocessing_tansforms,
            ]
        )
        return transforms


def get_autoencoder_transforms(
    data: str, out_shape: tuple = (112, 112, 112), ct_window: tuple = (500, 1500)
) -> tio.Compose:

    lower_hu_bound = ct_window[0] - int(ct_window[1] / 2.0)
    upper_hu_bound = ct_window[0] + int(ct_window[1] / 2.0)

    if data == "input":
        transforms = tio.Compose(
            [
                tio.transforms.Resize(out_shape),
                tio.transforms.Clamp(out_min=lower_hu_bound, out_max=upper_hu_bound),
                tio.transforms.RescaleIntensity(
                    out_min_max=(-1, 1), in_min_max=(lower_hu_bound, upper_hu_bound)
                ),
            ]
        )
    elif data == "target":
        transforms = tio.Compose(
            [
                tio.transforms.Resize(out_shape),
                tio.transforms.Clamp(out_min=lower_hu_bound, out_max=upper_hu_bound),
                tio.transforms.RescaleIntensity(
                    out_min_max=(0, 1), in_min_max=(lower_hu_bound, upper_hu_bound)
                ),
            ]
        )
    else:
        raise ValueError(
            'Invalid value for "data". Can only transform "input" or "target", but <{:s}> was given.'.format(
                data
            )
        )

    return transforms


def resample_sitk_image(
    sitk_image: sitk.Image, new_spacing: list[float] = [1.0, 1.0, 1.0]
) -> tuple[sitk.Image, np.ndarray]:
    """
    Resample SITK Image.
    """
    # Get information about the orginal image
    num_dim = sitk_image.GetDimension()
    orig_pixelid = sitk_image.GetPixelIDValue()
    orig_origin = sitk_image.GetOrigin()
    orig_direction = sitk_image.GetDirection()
    orig_spacing = np.array(sitk_image.GetSpacing())
    orig_size = np.array(sitk_image.GetSize(), dtype=np.int32)

    logger.debug("Orginal image properties:")
    logger.debug("num_dim", num_dim)
    logger.debug("orig_pixelid", orig_pixelid)
    logger.debug("orig_origin", orig_origin)
    logger.debug("orig_direction", orig_direction)
    logger.debug("orig_spacing", orig_spacing)
    logger.debug("orig_size", orig_size)

    # Calculate scaling factor
    scaling_factor = orig_spacing / new_spacing
    logger.debug("Scaling factor:", scaling_factor)

    # Calculate new size
    new_size = orig_size * scaling_factor
    new_size = np.ceil(new_size).astype(np.int32)
    new_size = [int(s) for s in new_size]

    # Define resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputOrigin(orig_origin)
    resampler.SetOutputDirection(orig_direction)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)

    # Perform resampling
    resampled_sitk_image = resampler.Execute(sitk_image)

    # Get information about the resampled image
    num_dim = resampled_sitk_image.GetDimension()
    orig_pixelid = resampled_sitk_image.GetPixelIDValue()
    orig_origin = resampled_sitk_image.GetOrigin()
    orig_direction = resampled_sitk_image.GetDirection()
    orig_spacing = np.array(resampled_sitk_image.GetSpacing())
    orig_size = np.array(resampled_sitk_image.GetSize(), dtype=np.int32)

    logger.debug("Resampled image properties:")
    logger.debug("num_dim", num_dim)
    logger.debug("orig_pixelid", orig_pixelid)
    logger.debug("orig_origin", orig_origin)
    logger.debug("orig_direction", orig_direction)
    logger.debug("orig_spacing", orig_spacing)
    logger.debug("orig_size", orig_size)

    return resampled_sitk_image, scaling_factor
