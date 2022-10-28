from abc import ABC, abstractmethod

from pytorch_lightning import LightningDataModule
from torch import nn


class ModelProvider(ABC):

    @classmethod
    @abstractmethod
    def get_provider(cls, **kwargs) -> 'ModelProvider':
        """Create/instantiate the 'ModelProvider' object."""
        ...

    @abstractmethod
    def get_model(self, checkpoint, litmodel_kwargs, checkpoint_kwargs) -> nn.Module:
        ...

    @abstractmethod
    def get_lightning_data_module(
            self, train_annotation_file: str, val_annotation_file: str, test_annotation_file: str,
            img_train_base_dir: str = None, img_val_base_dir: str = None,
            img_test_base_dir: str = None, batch_size: int = 1) -> LightningDataModule:
        ...
