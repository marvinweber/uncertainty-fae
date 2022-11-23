from abc import ABC, abstractmethod

from pytorch_lightning import LightningDataModule
from torch import nn


class ModelProvider(ABC):

    @classmethod
    @abstractmethod
    def get_provider(
        cls,
        train_config = None,
        eval_mode: bool = False,
        **kwargs
    ) -> 'ModelProvider':
        """Create/instantiate the 'ModelProvider' object."""
        ...

    @abstractmethod
    def get_model(self, checkpoint, litmodel_kwargs, checkpoint_kwargs) -> nn.Module:
        ...

    @abstractmethod
    def get_lightning_data_module(
        self,
        train_annotation_file: str,
        val_annotation_file: str,
        test_annotation_file: str,
        img_train_base_dir: str = None,
        img_val_base_dir: str = None,
        img_test_base_dir: str = None,
        batch_size: int = 1,
        num_workers: int = 4,
    ) -> LightningDataModule:
        """
        Get the Lightning Data Module for given train/val/test annotations and directories.

        Args:
            train_annotation_file: (CSV) annotation file for training samples.
            val_annotation_file: (CSV) annotation file for validation samples.
            test_annotation_file: (CSV) annotation file for test samples.
            img_train_base_dir: Base directory in which training samples are located (for dynamic/
                relative path usage in the annoation file).
            img_val_base_dir: Base directory of validation samples (cf. `img_train_base_dir`).
            img_test_base_dir: Base directory of test samples (cf. `img_train_base_dir`).
            batch_size: Batch size to use by the DataLoaders.
            num_workers: Amount of workers to use by the DataLoader.
        
        Returns:
            The `LightningDataModule` according to the given configuration.
        """
        ...