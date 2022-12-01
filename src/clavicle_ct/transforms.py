import torchio as tio


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
