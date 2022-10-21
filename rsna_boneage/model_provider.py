
from typing import Dict, Tuple

from pytorch_lightning import LightningDataModule, LightningModule
from torch import nn
from torchvision import transforms
from torchvision.models import inception_v3, resnet18, resnet34, resnet50

from rsna_boneage.litmodel.rsna import LitRSNABoneageMCDropout
from rsna_boneage.litmodel.rsna_variance import (LitRSNABoneageVarianceNet,
                                                 LitRSNABoneageVarianceNetMCDropout)
from util import ModelProvider

from .data import RSNABoneageDataModule
from .litmodel import LitRSNABoneage
from .net.inception import RSNABoneageInceptionNetWithGender
from .net.resnet import RSNABoneageResNetWithGender
from .net.resnet import resnet18 as boneage_resnet18
from .net.resnet import resnet34 as boneage_resnet34
from .net.resnet import resnet50 as boneage_resnet50

RSNA_LITMODEL_MAPPING: Dict[str, LightningModule] = {
    'base': LitRSNABoneage,
    'mc_dropout': LitRSNABoneageMCDropout,
    'deep_ensemble': None,
    'laplace_approx': None,
    'swag': None,
}

RSNA_VARIANCE_LITMODEL_MAPPING: Dict[str, LightningModule] = {
    'base': LitRSNABoneageVarianceNet,
    'mc_dropout': LitRSNABoneageVarianceNetMCDropout,
    'deep_ensemble': None,
    'laplace_approx': None,
    'swag': None,
}


class RSNAModelProvider(ModelProvider):

    def __init__(self, base_net: str, uncertainty_method: str,
                 img_input_dimensions: Tuple[int, int], variance_net: bool, with_gender_input: bool,
                 rescale_boneage: bool = True, rebalance_classes: bool = True,
                 with_pretrained_weights: bool = True) -> None:
        super().__init__()
        self.base_net = base_net
        self.uncertainty_method = uncertainty_method
        self.img_input_dimensions = tuple(img_input_dimensions)
        self.variance_net = variance_net
        self.with_gender_input = with_gender_input
        self.rescale_boneage = rescale_boneage
        self.rebalance_classes = rebalance_classes
        self.with_pretrained_weights = with_pretrained_weights

    def get_model(self, checkpoint=None, litmodel_kwargs=dict()) -> nn.Module:
        # Create Network
        output_neurons = 2 if self.variance_net else 1
        if self.base_net == 'inceptionv3':
            net = _get_inception(self.with_gender_input, self.with_pretrained_weights,
                                 output_neurons)
        elif self.base_net in ['resnet18', 'resnet34', 'resnet50']:
            net = _get_resnet(self.base_net, self.with_gender_input, self.with_pretrained_weights,
                              output_neurons)
        else:
            raise ValueError(f'Invalid Network / Model Name! ({self.base_net})')

        # Get Litmodel Class
        litmodel_mapping = (RSNA_VARIANCE_LITMODEL_MAPPING
                            if self.variance_net
                            else RSNA_LITMODEL_MAPPING)
        litmodel_cls = litmodel_mapping[self.uncertainty_method]

        # Create Litmodel
        if checkpoint is not None:
            litmodel = litmodel_cls.load_from_checkpoint(checkpoint, net=net, **litmodel_kwargs)
        else:
            litmodel = litmodel_cls(net=net, **litmodel_kwargs)

        return litmodel

    def get_lightning_data_module(
            self, train_annotation_file: str, val_annotation_file: str, test_annotation_file: str,
            img_train_base_dir: str = None, img_val_base_dir: str = None,
            img_test_base_dir: str = None, batch_size: int = 8) -> LightningDataModule:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.5)
        ])
        datamodule = RSNABoneageDataModule(
            annotation_file_train=train_annotation_file,
            annotation_file_val=val_annotation_file,
            annotation_file_test=test_annotation_file,
            img_train_base_dir=img_train_base_dir,
            img_val_base_dir=img_val_base_dir,
            img_test_base_dir=img_test_base_dir,
            batch_size=batch_size,
            transform=transform,
            target_dimensions=self.img_input_dimensions,
            rescale_boneage=self.rescale_boneage,
            rebalance_classes=self.rebalance_classes,
            with_gender_input=self.with_gender_input,
        )

        return datamodule

    @classmethod
    def get_provider(cls, **kwargs) -> 'ModelProvider':
        return RSNAModelProvider(**kwargs)


# def get_model_and_datamodule(
#             name: str, img_input_dimension: tuple, with_gender_input: bool, rescale_boneage=True,
#             rebalance_classes=True, with_pretrained_weights_if_avail=True, batch_size=8,
#             variance_network=False, checkpoint_path: str = None, litmodel_kwargs: dict = dict()
#         ) -> Tuple[RSNABoneageLitModel, RSNABoneageDataModule]:
#     # TODO: Annotation File Path configurable!
#     net = None
#     output_neurons = 2 if variance_network else 1
#     transform = transforms.Compose([
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(30),
#         transforms.ColorJitter(brightness=0.5)
#     ])
#     datamodule = RSNABoneageDataModule(
#         annotation_file_train='/mnt/datassd/data_fae_uncertainty/train_annotations_preprocessed_500x500.csv',
#         annotation_file_val='/mnt/datassd/data_fae_uncertainty/val_annotations_preprocessed_500x500.csv',
#         annotation_file_test='/mnt/datassd/data_fae_uncertainty/test_annotations_preprocessed_500x500.csv',
#         batch_size=batch_size,
#         transform=transform,
#         target_dimensions=img_input_dimension,
#         rescale_boneage=rescale_boneage,
#         rebalance_classes=rebalance_classes,
#         with_gender_input=with_gender_input)

#     # INCEPTION
#     if name == 'inceptionv3':
#         net = _get_inception(with_gender_input, with_pretrained_weights_if_avail, output_neurons)

#     # RESNETS
#     elif name in ['resnet18', 'resnet34', 'resnet50']:
#         net = _get_resnet(name, with_gender_input, with_pretrained_weights_if_avail, output_neurons)

#     else:
#         raise ValueError(f'Invalid Network / Model Name! ({name})')

#     litmodel_class: LightningModule = (RSNABoneageVarianceLitModel 
#                                        if variance_network
#                                        else RSNABoneageLitModel)
#     # Either create "empty" LitModel or load from given Checkpoint file
#     if checkpoint_path is not None:
#         litmodel = litmodel_class.load_from_checkpoint(checkpoint_path, net=net, **litmodel_kwargs)
#     else:
#         litmodel = litmodel_class(net=net, **litmodel_kwargs)

#     return litmodel, datamodule


def _get_inception(with_gender_input: bool, with_pretrained_weights_if_avail=True,
                   output_neurons=1) -> nn.Module:
    inception_num_classes=output_neurons if not with_gender_input else 1000
    inception_net = inception_v3(pretrained=False, num_classes=inception_num_classes)

    if with_pretrained_weights_if_avail:
        inception_pretrained = inception_v3(pretrained=True, progress=True)
        inception_pretrained_state_dict = inception_pretrained.state_dict()

        inception_state_dict = inception_net.state_dict()
        for key in inception_state_dict.keys():
            if key in inception_pretrained_state_dict and 'fc' not in key:
                inception_state_dict[key] = inception_pretrained_state_dict[key]
        inception_net.load_state_dict(inception_state_dict)

    if with_gender_input:
        return RSNABoneageInceptionNetWithGender(
            inception=inception_net, output_neurons=output_neurons)
    else:
        return inception_net


def _get_resnet(name: str, with_gender_input: bool,
                with_pretrained_weights_if_avail=True, output_neurons=1) -> nn.Module:
    resnet = None
    resnet_pretrained = None
    num_classes_resnet = output_neurons if not with_gender_input else 1000

    if name == 'resnet18':
        resnet = boneage_resnet18(pretrained=False, progress=True, num_classes=num_classes_resnet)
        resnet_pretrained = resnet18(pretrained=True, progress=True)
    elif name == 'resnet34':
        resnet = boneage_resnet34(pretrained=False, progress=True, num_classes=num_classes_resnet)
        resnet_pretrained = resnet34(pretrained=True, progress=True)
    elif name == 'resnet50':
        resnet = boneage_resnet50(pretrained=False, progress=True, num_classes=num_classes_resnet)
        resnet_pretrained = resnet50(pretrained=True, progress=True)

    if resnet is None:
        raise ValueError('Invalid Arguments!')

    if with_pretrained_weights_if_avail:
        resnet_pretrained_state_dict = resnet_pretrained.state_dict()
        resnet_state_dict = resnet.state_dict()

        for key in resnet_state_dict.keys():
            if key in resnet_pretrained_state_dict and 'fc' not in key:
                resnet_state_dict[key] = resnet_pretrained_state_dict[key]
        resnet.load_state_dict(resnet_state_dict)

    if with_gender_input:
        return RSNABoneageResNetWithGender(resnet=resnet, output_neurons=output_neurons)
    else:
        return resnet
