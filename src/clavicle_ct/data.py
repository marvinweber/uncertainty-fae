import os
from pathlib import Path
import re
from typing import Optional

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torchio as tio
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from clavicle_ct.transforms import get_transforms
from uncertainty_fae.util import TensorList

CLAVICLE_CT_DATASET_MIN_AGE = 5478.0  # = 15 x 365.25
CLAVICLE_CT_DATASET_MAX_AGE = 10958.0  # = 30 x 365.25


class CTDataset(Dataset):
    """
    CT dataset.

    Attributes
    ----------
        annotation_file : string
            Path to a csv file with annotations
        transforms : list (optional)
            Optional list of transforms to be applied to a sample
    """

    def __init__(
        self,
        annotation_file: str,
        batch_size: int,
        img_base_dir: Optional[str] = None,
        include_sex: bool = False,
        use_full_ct: bool = False,
        full_ct_image_base_dir: Optional[str] = None,
        cropping: str = "random",
        transforms_input: Optional[tio.Compose] = None,
        transforms_target: Optional[tio.Compose] = None,
    ) -> None:
        super().__init__()

        self.annotation_file = annotation_file
        self.img_base_dir = img_base_dir
        self.batch_size = batch_size
        self.include_sex = include_sex
        self.use_full_ct = use_full_ct
        self.full_ct_image_base_dir = full_ct_image_base_dir
        self.cropping = cropping
        self.transforms_input = transforms_input
        self.transforms_target = transforms_target

        self.annotations = pd.read_csv(annotation_file)
        if self.use_full_ct:
            if not self.full_ct_image_base_dir:
                raise ValueError("Cannot use full-CT wihtout `full_ct_image_base_dir`!")
            self.annotations["image"] = self._use_original_images(
                self.annotations["image"].to_list()
            )

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx) -> tuple:
        """
        Typically, a Dataset returns a single sample. However, this Dataset will return a batch of samples.
        The corresponding DataLoader is adjusted accordingly, by providing a 'collate_fn' function.
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image filepath
        image_file = str(self.annotations.loc[idx, "image"])

        # Adjust filepath if necessary
        if self.img_base_dir:
            image_file = os.path.join(self.img_base_dir, image_file)

        # Load image from disk
        try:
            if self.use_full_ct:
                image = sitk.ReadImage(image_file)
                image = sitk.GetArrayFromImage(image)
                image = image.astype(np.float32)
            else:
                image = np.load(image_file)
                image = image.astype(np.float32)
        except:
            raise ValueError("Cannot load image <{:s}>".format(image_file))

        # Get image dimensions
        x, y, z = image.shape[0], image.shape[1], image.shape[2]

        # Prepare cropping
        patch_size = 112

        if self.cropping == "random":
            rng = np.random.default_rng()
        elif self.cropping == "fixed":
            rng = np.random.default_rng(seed=0)
        else:
            raise ValueError(
                '"cropping" has to be set to "random" or "fixed", got "{:s}"'.format(self.cropping)
            )

        # Fill batch with patches from the same image to increase performance
        X_batch = []
        y_batch = []

        for i in range(self.batch_size):

            x0, x1 = 0, x
            y0, y1 = 0, y
            z0, z1 = 0, z

            # Generate patch position
            if x > patch_size:
                upper_x0_bound = x - patch_size
                x0 = rng.integers(low=0, high=upper_x0_bound, endpoint=True)
                x1 = x0 + patch_size
            if y > patch_size:
                upper_y0_bound = y - patch_size
                y0 = rng.integers(low=0, high=upper_y0_bound, endpoint=True)
                y1 = y0 + patch_size
            if z > patch_size:
                upper_z0_bound = z - patch_size
                z0 = rng.integers(low=0, high=upper_z0_bound, endpoint=True)
                z1 = z0 + patch_size

            # Do cropping
            patch = image[x0:x1, y0:y1, z0:z1]

            # Prepare dimensions and set up tensors
            patch = np.expand_dims(patch, 0)  # Add color channel dimension
            X_patch = torch.from_numpy(patch)
            y_patch = torch.from_numpy(patch)

            # Apply transforms to input and target image
            if self.transforms_input:
                X_patch = self.transforms_input(X_patch)
            if self.transforms_target:
                y_patch = self.transforms_target(y_patch)

            # Add batch dimension. The DataLoader will use this dimension
            # to concatenate all patches inside the batch list
            X_patch = torch.unsqueeze(X_patch, dim=0)
            y_patch = torch.unsqueeze(y_patch, dim=0)

            # Add patch to batch list
            X_batch.append(X_patch)
            y_batch.append(y_patch)

        # Fill metadata batch (always the same because the image is also the same)
        z_batch = []
        if self.include_sex:

            sex = self.annotations.loc[idx, "sex"]

            if sex == "M":
                sex = torch.tensor([0.0], dtype=torch.float32)
            elif sex == "F":
                sex = torch.tensor([1.0], dtype=torch.float32)
            else:
                raise ValueError('Sex must be "M" or "F". Got <{}>.'.format(sex))

            # Add batch dimension. The DataLoader will use this dimension
            # to concatenate all patches inside the batch list
            sex = torch.unsqueeze(sex, dim=0)

            z_batch = [sex] * self.batch_size

        if not self.include_sex:
            return X_batch, y_batch
        else:
            return X_batch, y_batch, z_batch

    def _use_original_images(self, image_files) -> list:
        """Replace filepaths for preprocessed images with original images."""
        new_image_files = []

        for old_filename in image_files:
            _, _, _, full_ct_file = get_patient_pseudonyms_from_ct_name(old_filename)
            full_ct_filepath = os.path.join(self.full_ct_image_base_dir, full_ct_file)
            new_image_files.append(full_ct_filepath)

        return new_image_files


class ClavicleDataset(Dataset):
    """
    Clavicle CT Dataset.

    Attributes
    ----------
        annotation_file : string
            Path to a csv file with annotations
        transforms : list (optional)
            Optional list of transforms to be applied to a sample
    """

    def __init__(
        self,
        annotation_file: str,
        img_base_dir: Optional[str] = None,
        transforms: Optional[tio.Compose] = None,
        rescale_age: bool = False,
        rebalance_classes: bool = False,
        include_sex: bool = False,
    ) -> None:
        super().__init__()

        self.annotation_file = annotation_file
        self.img_base_dir = img_base_dir
        self.transforms = transforms
        self.rescale_age = rescale_age
        self.rebalance_classes = rebalance_classes
        self.include_sex = include_sex

        self.orig_annotations = pd.read_csv(annotation_file)
        self.annotations = self.orig_annotations.copy(deep=True)
        if self.rebalance_classes:
            self.annotations = self._sample_flat_age_distribution(self.annotations)

    def __len__(self):
        return self.annotations.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image filename and target
        image_file = str(self.annotations.loc[idx, "image"])
        age = self.annotations.loc[idx, "age"]
        sex = self.annotations.loc[idx, "sex"]

        # Adjust filepath if necessary
        if self.img_base_dir:
            image_file = os.path.join(self.img_base_dir, image_file)

        if not os.path.isfile(image_file):
            raise ValueError(f"Image not found: {image_file}!")

        # Load image from disk
        image = np.load(image_file).astype(np.float32)
        image = np.expand_dims(image, 0)  # Add channel dimension at the front
        image = torch.from_numpy(image)

        # Apply transforms
        if self.transforms:
            image = self.transforms(image)

        # Prepare age
        if self.rescale_age:
            age = clavicle_age_rescale(age)
        age = torch.tensor(age, dtype=torch.float32)

        if not self.include_sex:
            return image, age
        else:
            # Convert sex from string to float
            if sex == "M":
                sex = np.float32(0)
            elif sex == "F":
                sex = np.float32(1)
            else:
                raise ValueError(f"Unkown sex: '{sex}'!")

            # Prepare sex
            sex = np.expand_dims(sex, 0)  # Add batch dimension at the front
            sex = torch.from_numpy(sex)
            return TensorList([image, sex]), age

    def _sample_flat_age_distribution(self, annotations: pd.DataFrame) -> pd.DataFrame:
        """
        Samples a flat distribution

        Note: Sampling is done with replacement and therefore alters the original data distribution.
        """
        # Define age bins (a column for sex bins already exists)
        age_bins = np.linspace(start=15.0, stop=30.0, num=16, endpoint=True, dtype=np.int32)

        # Add column with age in years
        annotations["age_years"] = annotations["age"].to_numpy() / 365.25

        # Bin data based on age
        annotations["age_bin"] = np.digitize(annotations["age_years"].to_numpy(), age_bins)

        # Identify number of samples to be drawn per bin, i.e. per age and sex
        _, counts = np.unique(annotations["age_bin"].to_numpy(), return_counts=True)
        n_samples_per_bin = int(np.max(counts) / 2.0)

        # Sample a selected number of times from each bin
        annotations = (
            annotations.groupby(["age_bin", "sex"])
            .apply(lambda x: x.sample(n_samples_per_bin, replace=True))
            .reset_index(drop=True)
        )

        return annotations

    def reset_annotations(self) -> None:
        if self.rebalance_classes:
            self.annotations = self._sample_flat_age_distribution(self.orig_annotations)
        else:
            self.annotations = self.orig_annotations


class CTDataModule(LightningDataModule):
    def __init__(
        self,
        annotation_file_train: str,
        annotation_file_val: str,
        img_train_base_dir: str = None,
        img_val_base_dir: str = None,
        batch_size: int = 1,
        include_sex: bool = False,
        cropping: str = "random",
        transforms_input: Optional[tio.Compose] = None,
        transforms_target: Optional[tio.Compose] = None,
        num_workers=12,
        full_ct_image_base_dir: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.annotation_file_train = annotation_file_train
        self.annotation_file_val = annotation_file_val
        self.img_train_base_dir = img_train_base_dir
        self.img_val_base_dir = img_val_base_dir
        self.batch_size = batch_size
        self.include_sex = include_sex
        self.cropping = cropping
        self.transforms_input = transforms_input
        self.transforms_target = transforms_target
        self.num_workers = num_workers
        self.full_ct_image_base_dir = full_ct_image_base_dir

    def setup(self, stage: str) -> None:
        if stage == "fit" or "validate":
            self.dataset_train = CTDataset(
                annotation_file=self.annotation_file_train,
                img_base_dir=self.img_train_base_dir,
                batch_size=self.batch_size,
                include_sex=self.include_sex,
                cropping=self.cropping,
                transforms_input=self.transforms_input,
                transforms_target=self.transforms_target,
                full_ct_image_base_dir=self.full_ct_image_base_dir,
            )
            self.dataset_val = CTDataset(
                annotation_file=self.annotation_file_val,
                img_base_dir=self.img_val_base_dir,
                batch_size=self.batch_size,
                include_sex=self.include_sex,
                cropping=self.cropping,
                transforms_input=self.transforms_input,
                transforms_target=self.transforms_target,
                full_ct_image_base_dir=self.full_ct_image_base_dir,
            )
        else:
            raise ValueError(
                "This DataModule is for training and validation only and not for testing!"
            )

    def collate_fn(self, data):
        if not self.include_sex:
            X_batch, y_batch = data[0]
        else:
            X_batch, y_batch, z_batch = data[0]

        X_batch = torch.cat(X_batch)
        y_batch = torch.cat(y_batch)
        if self.include_sex:
            z_batch = torch.cat(z_batch)

        if not self.include_sex:
            return X_batch, y_batch
        else:
            return X_batch, y_batch, z_batch

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=1,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=True,
        )


class ClavicleDataModule(LightningDataModule):
    def __init__(
        self,
        annotation_file_train: str,
        annotation_file_val: str,
        annotation_file_test: str,
        img_train_base_dir: str = None,
        img_val_base_dir: str = None,
        img_test_base_dir: str = None,
        batch_size: int = 8,
        transforms_train: Optional[tio.Compose] = None,
        shuffle_train: bool = True,
        rescale_age: bool = False,
        rebalance_classes: bool = False,
        with_sex_input: bool = False,
        num_workers: int = 8,
    ) -> None:
        super().__init__()

        self.annotation_file_train = annotation_file_train
        self.annotation_file_val = annotation_file_val
        self.annotation_file_test = annotation_file_test
        self.img_train_base_dir = img_train_base_dir
        self.img_val_base_dir = img_val_base_dir
        self.img_test_base_dir = img_test_base_dir
        self.batch_size = batch_size
        self.transforms_train = transforms_train
        self.shuffle_train = shuffle_train
        self.rescale_age = rescale_age
        self.rebalance_classes = rebalance_classes
        self.with_sex_input = with_sex_input
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        # Training Dataset
        if stage in ["fit", "train", "all", ""]:
            if not self.transforms_train:
                self.transforms_train = get_transforms(augmentation=False)
            self.dataset_train = ClavicleDataset(
                annotation_file=self.annotation_file_train,
                img_base_dir=self.img_train_base_dir,
                transforms=self.transforms_train,
                rescale_age=self.rescale_age,
                rebalance_classes=self.rebalance_classes,
                include_sex=self.with_sex_input,
            )

        # Validation Dataset
        if stage in ["val", "validate", "fit", "all", ""]:
            self.dataset_val = ClavicleDataset(
                annotation_file=self.annotation_file_val,
                img_base_dir=self.img_val_base_dir,
                transforms=get_transforms(augmentation=False),
                rescale_age=self.rescale_age,
                include_sex=self.with_sex_input,
            )

        # Test Dataset
        if stage in ["test", "all", ""]:
            self.dataset_test = ClavicleDataset(
                annotation_file=self.annotation_file_test,
                img_base_dir=self.img_test_base_dir,
                transforms=get_transforms(augmentation=False),
                rescale_age=self.rescale_age,
                include_sex=self.with_sex_input,
            )

    def train_dataloader(self):
        self.dataset_train.reset_annotations()
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


def get_patient_pseudonyms_from_ct_name(filename: str) -> tuple[int, int, int, str]:
    """
    Extract patient, study, and series from given (preprocessed) file name.

    Args:
        filename: Filename of the preprocessed image, from wich info should be extracted.
    
    Returns:
        A tuple `(patient, study, series, full_ct_file)` where the first three integers correspond
        to the patient, study and series pseudonyms, while `full_ct_file` corresponds to the
        Nifiti image path (with directory structure).
    """
    filename = Path(filename).stem
    regex = re.compile(r"ae_(?P<patient>\d+)_(?P<study>\d+)_(?P<series>\d+)")
    res = regex.search(filename)
    patient, study, series = [
        int(res.group("patient")), int(res.group("study")), int(res.group("series"))
    ]

    full_ct_file = "ae_{:n}/ae_{:n}/ae_{:n}.nii.gz".format(patient, study, series)
    return patient, study, series, full_ct_file


def clavicle_age_rescale(age):
    """Scale given age to values between [0, 1] according to train data."""
    lower_bound = CLAVICLE_CT_DATASET_MIN_AGE
    upper_bound = CLAVICLE_CT_DATASET_MAX_AGE
    return (age - lower_bound) / (upper_bound - lower_bound)


def undo_clavicle_age_rescale(age_rescaled, with_shift: bool = True):
    """Undo (down) scaling of age.

    Args:
        age_rescaled: The rescaled age (e.g., prediction or target (tensor)).
        with_shift: Whether only the relative magnitude should be re-created (`false`, e.g., scale
            from [0,1] to years), or if the lower bound of the original data-range should be added,
            as well (`true`, default).
    """
    lower_bound = CLAVICLE_CT_DATASET_MIN_AGE if with_shift else 0
    upper_bound = CLAVICLE_CT_DATASET_MAX_AGE
    return (age_rescaled * (upper_bound - lower_bound)) + lower_bound
