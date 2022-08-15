import torch
from torch.utils.data import DataLoader, Dataset, random_split

from torchvision import transforms
from pytorch_lightning import LightningDataModule

from PIL import Image
import numpy as np
import pandas as pd
from  typing import Optional


class RSNABoneageDataset(Dataset):
    """
    RSNA Boneage Dataset.
    
    Attributes
    ----------
        annotation_file : string
            Path to a csv file with annotations
        transform : list (optional)
            Optional list of transforms to be applied to a sample
    """
    def __init__(self, annotation_file: str, transform=None,
                 target_dimensions=(200, 200), rescale_boneage=False,
                 rebalance_classes=False) -> None:
        super().__init__()

        self.annotation_file = annotation_file
        self.annotations = pd.read_csv(annotation_file)

        if rebalance_classes:
            # Divide age into categories / bins (10 bins)
            self.annotations['boneage_category'] = pd.cut(self.annotations['boneage'], 10)
            # Convert male to int value
            self.annotations['male_numeric'] = self.annotations.apply(
                lambda row: 1 if row[['male']].bool() else 0, axis=1)
            # ensure bone category/bin and male groups contain each 1100 images
            self.annotations = self.annotations.groupby(['boneage_category', 'male_numeric']).apply(
                lambda x: x.sample(1100, replace = True)).reset_index(drop=True)

        self.transform = transform
        self.target_dimensions = target_dimensions
        self.rescale_boneage = rescale_boneage

    def __len__(self):
        return self.annotations.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image filename and target
        img_path = self.annotations.loc[idx, 'img_path']
        img_preprocessed_path = self.annotations.loc[idx, 'img_preprocessed_path']
        boneage = self.annotations.loc[idx, 'boneage']
        boneage = torch.tensor(np.float32(boneage))

        # Load image from disk
        if not img_preprocessed_path or np.isnan(img_preprocessed_path):
            image = Image.open(img_path).convert('RGB')
        else:
            image: torch.Tensor = torch.load(img_preprocessed_path)
            image_pil: Image.Image = transforms.ToPILImage()(image)
            image = image_pil.convert('RGB')

        # Apply additional transforms
        if self.transform:
            image = self.transform(image)

        # Resize and convert to Tensor
        transform = get_image_transforms(target_dimensions=self.target_dimensions)
        image: torch.Tensor = transform(image)

        if self.rescale_boneage:
            boneage = self._rescale_boneage(boneage)

        return image, boneage

    def _rescale_boneage(self, boneage):
        lower_bound = 0  # 0 months
        upper_bound = 230  # 230 months
        return (boneage - lower_bound) / (upper_bound - lower_bound)
    
    def get_bonage_distribution(self, bins: int = 25) -> dict[pd.Interval, int]:
        annotations = self.annotations.copy(deep=True)
        annotations['boneage_unscaled'] = annotations['boneage']
        if self.rescale_boneage:
            annotations['boneage'] = annotations['boneage'].map(lambda b: self._rescale_boneage(b))
        
        age_bins = pd.cut(annotations['boneage'], bins=bins)
        age_grouped = annotations.groupby(age_bins).count()
        return dict(zip(age_grouped.index.to_list(), age_grouped['boneage'].tolist()))


class RSNABoneageDataModule(LightningDataModule):
    def __init__(self, 
                 annotation_file_train: str,
                 annotation_file_val: str,
                 annotation_file_test: str,
                 batch_size: int = 1,
                 transform=None,
                 target_dimensions=(200, 200),
                 rescale_boneage=False,
                 rebalance_classes=False,
                 split_seed: Optional[int] = None) -> None:
        super().__init__()

        self.annotation_file_train = annotation_file_train
        self.annotation_file_val = annotation_file_val
        self.annotation_file_test = annotation_file_test
        self.batch_size = batch_size
        self.transform  = transform
        self.split_seed = split_seed
        self.target_dimensions = target_dimensions
        self.rescale_boneage = rescale_boneage
        self.rebalance_classes = rebalance_classes

    def setup(self, stage: str) -> None:
        # Training Dataset
        self.dataset_train = RSNABoneageDataset(
            annotation_file=self.annotation_file_train, transform=self.transform,
            target_dimensions=self.target_dimensions, rescale_boneage=self.rescale_boneage,
            rebalance_classes=self.rebalance_classes)

        # Validation Dataset
        self.dataset_val = RSNABoneageDataset(
            annotation_file=self.annotation_file_val, transform=self.transform,
            target_dimensions=self.target_dimensions, rescale_boneage=self.rescale_boneage,
            rebalance_classes=self.rebalance_classes)

        # Test Dataset
        self.dataset_test = RSNABoneageDataset(
            annotation_file=self.annotation_file_test, transform=self.transform,
            target_dimensions=self.target_dimensions, rescale_boneage=self.rescale_boneage)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size,
                          shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size,
                          shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size,
                          shuffle=False, num_workers=4)


def get_image_transforms(target_dimensions=(200, 200)):
    return transforms.Compose([
        transforms.Resize(target_dimensions),
        transforms.ToTensor(),
    ])
