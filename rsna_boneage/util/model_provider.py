
from typing import Tuple

from torch import nn
from torchvision import transforms
from torchvision.models import inception_v3, resnet18, resnet34, resnet50

from boneage.rsna_bone_dataloading import RSNABoneageDataModule
from boneage.rsna_bone_litmodel import RSNABoneageLitModel
from boneage.rsna_boneage_resnet import RSNABoneageResNetWithGender
from boneage.rsna_boneage_resnet import resnet18 as boneage_resnet18
from boneage.rsna_boneage_resnet import resnet34 as boneage_resnet34
from boneage.rsna_boneage_resnet import resnet50 as boneage_resnet50
from boneage.rsna_boneage_inception import RSNABoneageInceptionNetWithGender


def get_model_and_datamodule(
            name: str, img_input_dimension: tuple, with_gender_input: bool, rescale_boneage=True,
            rebalance_classes=True, with_pretrained_weights_if_avail=True, batch_size=8,
            checkpoint_path: str = None, litmodel_kwargs: dict = dict()
        ) -> Tuple[RSNABoneageLitModel, RSNABoneageDataModule]:
    # TODO: Annotation File Path configurable!
    net = None
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.5)
    ])
    datamodule = RSNABoneageDataModule(
        annotation_file_train='/mnt/datassd/data_fae_uncertainty/train_annotations_preprocessed_500x500.csv',
        annotation_file_val='/mnt/datassd/data_fae_uncertainty/val_annotations_preprocessed_500x500.csv',
        annotation_file_test='/mnt/datassd/data_fae_uncertainty/test_annotations_preprocessed_500x500.csv',
        batch_size=batch_size,
        transform=transform,
        target_dimensions=img_input_dimension,
        rescale_boneage=rescale_boneage,
        rebalance_classes=rebalance_classes,
        with_gender_input=with_gender_input)

    # INCEPTION
    if name == 'inceptionv3':
        net = _get_inception(with_gender_input, with_pretrained_weights_if_avail)

    # RESNETS
    elif name in ['resnet18', 'resnet34', 'resnet50']:
        net = _get_resnet(name, with_gender_input, with_pretrained_weights_if_avail)

    else:
        raise ValueError(f'Invalid Network / Model Name! ({name})')

    # Either create "empty" LitModel or load from given Checkpoint file
    if checkpoint_path is not None:
        litmodel = RSNABoneageLitModel.load_from_checkpoint(
            checkpoint_path, net=net, **litmodel_kwargs)
    else:
        litmodel = RSNABoneageLitModel(net=net, **litmodel_kwargs)

    return litmodel, datamodule


def _get_inception(with_gender_input: bool, with_pretrained_weights_if_avail=True) -> nn.Module:
    num_classes=1 if not with_gender_input else 1000
    inception_net = inception_v3(pretrained=False, num_classes=num_classes)

    if with_pretrained_weights_if_avail:
        inception_pretrained = inception_v3(pretrained=True, progress=True)
        inception_pretrained_state_dict = inception_pretrained.state_dict()

        inception_state_dict = inception_net.state_dict()
        for key in inception_state_dict.keys():
            if key in inception_pretrained_state_dict and 'fc' not in key:
                inception_state_dict[key] = inception_pretrained_state_dict[key]
        inception_net.load_state_dict(inception_state_dict)

    if with_gender_input:
        return RSNABoneageInceptionNetWithGender(inception=inception_net)
    else:
        return inception_net


def _get_resnet(name: str, with_gender_input: bool,
                with_pretrained_weights_if_avail=True) -> nn.Module:
    resnet = None
    resnet_pretrained = None
    num_classes = 1 if not with_gender_input else 1000

    if name == 'resnet18':
        resnet = boneage_resnet18(pretrained=False, progress=True, num_classes=num_classes)
        resnet_pretrained = resnet18(pretrained=True, progress=True)
    elif name == 'resnet34':
        resnet = boneage_resnet34(pretrained=False, progress=True, num_classes=num_classes)
        resnet_pretrained = resnet34(pretrained=True, progress=True)
    elif name == 'resnet50':
        resnet = boneage_resnet50(pretrained=False, progress=True, num_classes=num_classes)
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
        return RSNABoneageResNetWithGender(resnet=resnet)
    else:
        return resnet
