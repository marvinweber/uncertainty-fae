from typing import List, Optional, Type

import torch
import torch.nn as nn
from torch import Tensor

from uncertainty_fae.util import TensorList

from ..litmodel.autoencoder import LitClavicleAutoencoder
from .autoencoder import (
    agenet14_3d_autoencoder,
    agenet18_3d_autoencoder,
    agenet18_3d_light_autoencoder,
)


def conv1x1(channels_in: int, channels_out: int, stride: int = 1) -> nn.Conv2d:
    """
    1x1x1 convolution
    """
    return nn.Conv2d(channels_in, channels_out, kernel_size=1, stride=stride, bias=False)


def conv1x1x1(channels_in: int, channels_out: int, stride: int = 1) -> nn.Conv3d:
    """
    1x1x1 convolution
    """
    return nn.Conv3d(channels_in, channels_out, kernel_size=1, stride=stride, bias=False)


def conv3x3(channels_in: int, channels_out: int, stride: int = 1) -> nn.Conv2d:
    """
    3x3 convolution with padding
    """
    return nn.Conv2d(
        channels_in,
        channels_out,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=1,
        bias=False,
        dilation=1,
    )


def conv3x3x3(channels_in: int, channels_out: int, stride: int = 1) -> nn.Conv3d:
    """
    3x3x3 convolution with padding
    """
    return nn.Conv3d(
        channels_in,
        channels_out,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=1,
        bias=False,
        dilation=1,
    )


class BuildingBlock_2D(nn.Module):
    """
    Building block for the 18-layer and 32-layer ResNet architecture
    """

    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        stride: int = 1,
        norm_layer=nn.BatchNorm2d,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.conv1 = conv3x3(channels_in, channels_out, stride)
        self.bn1 = norm_layer(channels_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(channels_out, channels_out)
        self.bn2 = norm_layer(channels_out)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:

        identity = x

        # Convolution 1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Convolution 2
        out = self.conv2(out)
        out = self.bn2(out)

        # Shortcut connection (identity or projection)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BuildingBlock_3D(nn.Module):
    """
    Building block for the 18-layer and 32-layer ResNet architecture
    """

    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        stride: int = 1,
        norm_layer=nn.BatchNorm3d,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.conv1 = conv3x3x3(channels_in, channels_out, stride)
        self.bn1 = norm_layer(channels_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(channels_out, channels_out)
        self.bn2 = norm_layer(channels_out)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:

        identity = x

        # Convolution 1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Convolution 2
        out = self.conv2(out)
        out = self.bn2(out)

        # Shortcut connection (identity or projection)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# --------------------------------------------------
# AgeNet 3D Short (for AgeNet14)
# --------------------------------------------------


class AgeNet_3D_Short(nn.Module):
    def __init__(
        self,
        block: Type[BuildingBlock_3D],
        blocks_per_stack: List[int],
        num_classes: int = 1,
        use_dropout: bool = True,
        use_sex: bool = False,
        sex_neurons: int = 16,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.use_dropout = use_dropout
        self.use_sex = use_sex
        self.sex_neurons = sex_neurons
        self._norm_layer = nn.BatchNorm3d

        norm_layer = self._norm_layer

        self.conv1 = nn.Conv3d(1, 64, kernel_size=5, stride=2, padding=2)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._stack_blocks(block, blocks_per_stack[0], 64, 64)
        self.layer2 = self._stack_blocks(block, blocks_per_stack[1], 64, 128, 2, norm_layer)
        self.layer3 = self._stack_blocks(block, blocks_per_stack[2], 128, 256, 2, norm_layer)
        self.conv2 = nn.Conv3d(256, 512, kernel_size=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        if self.use_dropout:
            self.dropout = nn.Dropout(p=0.2)

        self.fc1 = nn.Linear(512, self.num_classes if not self.use_sex else 512 - self.sex_neurons)

        if self.use_sex:
            self.sex_layer = nn.Linear(1, self.sex_neurons)
            self.fc2 = nn.Linear(512, self.num_classes)

    def _stack_blocks(
        self,
        block: Type[BuildingBlock_3D],
        n_blocks: int,
        channels_in: int,
        channels_out: int,
        stride: int = 1,
        norm_layer=nn.BatchNorm3d,
    ) -> nn.Sequential:

        # Configure downsampling for the first block in the stack
        downsample = None
        if stride == 2:
            downsample = nn.Sequential(
                conv1x1x1(channels_in, channels_out, stride), norm_layer(channels_out)
            )
        elif stride != 1:
            raise ValueError("Stride is expected to be 1 or 2, got {:d}".format(stride))

        # Stack building blocks
        layers = []

        # First block
        layers.append(block(channels_in, channels_out, stride, norm_layer, downsample))

        # Additional blocks (stride and downsample are no longer considered)
        for _ in range(1, n_blocks):
            layers.append(block(channels_out, channels_out))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:

        x = self.conv1(x)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if self.use_dropout:
            x = self.dropout(x)

        x = self.fc1(x)

        if self.use_sex and y is not None:
            x = self.relu(x)
            y = self.sex_layer(y)
            x = torch.cat((x, y), dim=1)
            x = self.fc2(x)

        return x


# --------------------------------------------------
# AgeNet 3D (for AgeNet18 and AgeNet34)
# --------------------------------------------------


class AgeNet_3D(nn.Module):
    def __init__(
        self,
        block: Type[BuildingBlock_3D],
        blocks_per_stack: List[int],
        num_classes: int = 1,
        use_dropout: bool = True,
        use_sex: bool = False,
        sex_neurons: int = 16,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.use_dropout = use_dropout
        self.use_sex = use_sex
        self.sex_neurons = sex_neurons
        self._norm_layer = nn.BatchNorm3d

        norm_layer = self._norm_layer

        self.conv1 = nn.Conv3d(1, 64, kernel_size=5, stride=2, padding=2)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._stack_blocks(block, blocks_per_stack[0], 64, 64)
        self.layer2 = self._stack_blocks(block, blocks_per_stack[1], 64, 128, 2, norm_layer)
        self.layer3 = self._stack_blocks(block, blocks_per_stack[2], 128, 256, 2, norm_layer)
        self.layer4 = self._stack_blocks(block, blocks_per_stack[3], 256, 512, 2, norm_layer)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        if self.use_dropout:
            self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(512, self.num_classes if not self.use_sex else 512 - self.sex_neurons)

        if self.use_sex:
            self.sex_layer = nn.Linear(1, self.sex_neurons)
            self.fc2 = nn.Linear(512, self.num_classes)

    def _stack_blocks(
        self,
        block: Type[BuildingBlock_3D],
        n_blocks: int,
        channels_in: int,
        channels_out: int,
        stride: int = 1,
        norm_layer=nn.BatchNorm3d,
    ) -> nn.Sequential:

        # Configure downsampling for the first block in the stack
        downsample = None
        if stride == 2:
            downsample = nn.Sequential(
                conv1x1x1(channels_in, channels_out, stride), norm_layer(channels_out)
            )
        elif stride != 1:
            raise ValueError("Stride is expected to be 1 or 2, got {:d}".format(stride))

        # Stack building blocks
        layers = []

        # First block
        layers.append(block(channels_in, channels_out, stride, norm_layer, downsample))

        # Additional blocks (stride and downsample are no longer considered)
        for _ in range(1, n_blocks):
            layers.append(block(channels_out, channels_out))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor | TensorList) -> Tensor:

        if self.use_sex:
            img_input, sex_input = x

        x = self.conv1(img_input)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if self.use_dropout:
            x = self.dropout(x)

        x = self.fc1(x)

        if self.use_sex and sex_input is not None:
            x = self.relu(x)
            y = self.sex_layer(sex_input)
            if self.use_dropout:
                y = self.dropout(y)
            x = torch.cat((x, y), dim=1)
            x = self.fc2(x)

        return x


# --------------------------------------------------
# AgeNet 3D Light (for AgeNet18 Light)
# --------------------------------------------------


class AgeNet_3D_Light(nn.Module):
    def __init__(
        self,
        block_3d: Type[BuildingBlock_3D],
        block_2d: Type[BuildingBlock_2D],
        blocks_per_stack: List[int],
        num_classes: int = 1,
        use_dropout: bool = True,
        use_sex: bool = False,
        sex_neurons: int = 16,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.use_dropout = use_dropout
        self.use_sex = use_sex
        self.sex_neurons = sex_neurons
        self._norm_layer_3d = nn.BatchNorm3d
        self._norm_layer_2d = nn.BatchNorm2d

        norm_layer = self._norm_layer_3d

        self.conv1 = nn.Conv3d(1, 64, kernel_size=5, stride=2, padding=2)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._stack_blocks_3d(block_3d, blocks_per_stack[0], 64, 64)
        self.layer2 = self._stack_blocks_3d(
            block_3d, blocks_per_stack[1], 64, 128, 2, self._norm_layer_3d
        )
        self.layer3 = self._stack_blocks_3d(
            block_3d, blocks_per_stack[2], 128, 256, 2, self._norm_layer_3d
        )
        self.shrink = nn.Conv3d(256, 256, kernel_size=(1, 1, 7))
        self.layer4 = self._stack_blocks_2d(
            block_2d, blocks_per_stack[3], 256, 512, 2, self._norm_layer_2d
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if self.use_dropout:
            self.dropout = nn.Dropout(p=0.2)

        self.fc1 = nn.Linear(512, self.num_classes if not self.use_sex else 512 - self.sex_neurons)

        if self.use_sex:
            self.sex_layer = nn.Linear(1, self.sex_neurons)
            self.fc2 = nn.Linear(512, self.num_classes)

    def _stack_blocks_3d(
        self,
        block: Type[BuildingBlock_3D],
        n_blocks: int,
        channels_in: int,
        channels_out: int,
        stride: int = 1,
        norm_layer=nn.BatchNorm3d,
    ) -> nn.Sequential:

        # Configure downsampling for the first block in the stack
        downsample = None
        if stride == 2:
            downsample = nn.Sequential(
                conv1x1x1(channels_in, channels_out, stride), norm_layer(channels_out)
            )
        elif stride != 1:
            raise ValueError("Stride is expected to be 1 or 2, got {:d}".format(stride))

        # Stack building blocks
        layers = []

        # First block
        layers.append(block(channels_in, channels_out, stride, norm_layer, downsample))

        # Additional blocks (stride and downsample are no longer considered)
        for _ in range(1, n_blocks):
            layers.append(block(channels_out, channels_out))

        return nn.Sequential(*layers)

    def _stack_blocks_2d(
        self,
        block: Type[BuildingBlock_2D],
        n_blocks: int,
        channels_in: int,
        channels_out: int,
        stride: int = 1,
        norm_layer=nn.BatchNorm2d,
    ) -> nn.Sequential:

        # Configure downsampling for the first block in the stack
        downsample = None
        if stride == 2:
            downsample = nn.Sequential(
                conv1x1(channels_in, channels_out, stride), norm_layer(channels_out)
            )
        elif stride != 1:
            raise ValueError("Stride is expected to be 1 or 2, got {:d}".format(stride))

        # Stack building blocks
        layers = []

        # First block
        layers.append(block(channels_in, channels_out, stride, norm_layer, downsample))

        # Additional blocks (stride and downsample are no longer considered)
        for _ in range(1, n_blocks):
            layers.append(block(channels_out, channels_out))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:

        x = self.conv1(x)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.shrink(x)
        x = torch.squeeze(x, 4)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if self.use_dropout:
            x = self.dropout(x)

        x = self.fc1(x)

        if self.use_sex and y is not None:
            x = self.relu(x)
            y = self.sex_layer(y)
            x = torch.cat((x, y), dim=1)
            x = self.fc2(x)

        return x


# --------------------------------------------------
# Pretrained weights
# --------------------------------------------------
def update_model_with_autoencoder_weights(
    model_dict: dict, autoencoder: str, pretrained_weights: str, sex_input: bool
) -> dict:
    """
    Initialize a model from a checkpoint, this includes loading the saved weights.
    Next, update a given 'model_dict' with weights from the checkpoint.
    Only weights from layers which exist in both models will be updated.
    """
    print("Load pretrained weights <{:s}>".format(pretrained_weights))
    if autoencoder == "agenet14_3d_autoencoder":
        pretrained_model = LitClavicleAutoencoder.load_from_checkpoint(
            checkpoint_path=pretrained_weights,
            net=agenet14_3d_autoencoder(use_sex=False),
        )
    elif autoencoder == "agenet18_3d_autoencoder":
        pretrained_model = LitClavicleAutoencoder.load_from_checkpoint(
            checkpoint_path=pretrained_weights,
            net=agenet18_3d_autoencoder(use_sex=False),
        )
    elif autoencoder == "agenet18_3d_light_autoencoder":
        pretrained_model = LitClavicleAutoencoder.load_from_checkpoint(
            checkpoint_path=pretrained_weights,
            net=agenet18_3d_light_autoencoder(use_sex=False),
        )
    else:
        raise ValueError("Autoencoder name <{:s}> not accepted.".format(autoencoder))

    # Get model dictionary pretrained models
    pretrained_dict = pretrained_model.state_dict()

    # Filter out unnecessary keys
    decoder_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    if not sex_input:
        # fc1 layer is dependent on sex input
        decoder_dict.pop("net.fc1.weight")
        decoder_dict.pop("net.fc1.bias")

    # Overwrite model weights with pretrained weights
    model_dict.update(decoder_dict)

    return model_dict


# --------------------------------------------------
# Wrapper functions
# --------------------------------------------------


def _agenet_3d_short(
    block: Type[BuildingBlock_3D],
    blocks_per_stack: List[int],
    num_classes: int,
    use_dropout: bool,
    use_sex: bool,
    weights: Optional[str],
) -> AgeNet_3D_Short:

    model = AgeNet_3D_Short(block, blocks_per_stack, num_classes, use_dropout, use_sex)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict())

    return model


def _agenet_3d(
    block: Type[BuildingBlock_3D],
    blocks_per_stack: List[int],
    num_classes: int,
    use_dropout: bool,
    use_sex: bool,
    weights: Optional[str],
) -> AgeNet_3D:

    model = AgeNet_3D(block, blocks_per_stack, num_classes, use_dropout, use_sex)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict())

    return model


def _agenet_3d_light(
    block_3d: Type[BuildingBlock_3D],
    block_2d: Type[BuildingBlock_2D],
    blocks_per_stack: List[int],
    num_classes: int,
    use_dropout: bool,
    use_sex: bool,
    weights: Optional[str],
) -> AgeNet_3D_Light:

    model = AgeNet_3D_Light(block_3d, block_2d, blocks_per_stack, num_classes, use_dropout, use_sex)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict())

    return model


def agenet14_3d(
    *,
    num_classes: int = 1,
    use_dropout: bool = False,
    use_sex: bool = False,
    weights: Optional[str] = None
) -> AgeNet_3D_Short:
    """
    Convolutional network to predict age.

    The AgeNet18_3D model is adapted from the original ResNet18
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Parameters
    ----------
    blocks_per_stack : list
        Number of building blocks in each layer
    num_classes: int
        Number of output classes and neurons
    use_sex: bool
        Whether or not concatenate the last fully connected layer with a tabular feature that
        encodes sex. Also adds another fully connected layer.
    weights : str
        The pretrained weights for the model
    """
    return _agenet_3d_short(
        block=BuildingBlock_3D,
        blocks_per_stack=[2, 2, 2],
        num_classes=num_classes,
        use_dropout=use_dropout,
        use_sex=use_sex,
        weights=weights,
    )


def agenet18_3d(
    *,
    num_classes: int = 1,
    use_dropout: bool = False,
    use_sex: bool = False,
    weights: Optional[str] = None
) -> AgeNet_3D:
    """
    Convolutional network to predict age.

    The AgeNet18_3D model is adapted from the original ResNet18
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Parameters
    ----------
    blocks_per_stack : list
        Number of building blocks in each layer
    num_classes: int
        Number of output classes and neurons
    use_sex: bool
        Whether or not concatenate the last fully connected layer with a tabular feature that
        encodes sex. Also adds another fully connected layer.
    weights : str
        The pretrained weights for the model
    """
    return _agenet_3d(
        block=BuildingBlock_3D,
        blocks_per_stack=[2, 2, 2, 2],
        num_classes=num_classes,
        use_dropout=use_dropout,
        use_sex=use_sex,
        weights=weights,
    )


def agenet18_3d_light(
    *,
    num_classes: int = 1,
    use_dropout: bool = False,
    use_sex: bool = False,
    weights: Optional[str] = None
) -> AgeNet_3D_Light:
    """
    Convolutional network to predict age.

    The AgeNet18_3D model is adapted from the original ResNet18
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Parameters
    ----------
    blocks_per_stack : list
        Number of building blocks in each layer
    num_classes: int
        Number of output classes and neurons
    use_sex: bool
        Whether or not concatenate the last fully connected layer with a tabular feature that
        encodes sex. Also adds another fully connected layer.
    weights : str
        The pretrained weights for the model
    """
    return _agenet_3d_light(
        block_3d=BuildingBlock_3D,
        block_2d=BuildingBlock_2D,
        blocks_per_stack=[2, 2, 2, 2],
        num_classes=num_classes,
        use_dropout=use_dropout,
        use_sex=use_sex,
        weights=weights,
    )


def agenet34_3d(
    *,
    num_classes: int = 1,
    use_dropout: bool = False,
    use_sex: bool = False,
    weights: Optional[str] = None
) -> AgeNet_3D:
    """
    Convolutional network to predict age.

    The AgeNet18_3D model is adapted from the original ResNet18
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Parameters
    ----------
    blocks_per_stack : list
        Number of building blocks in each layer
    num_classes: int
        Number of output classes and neurons
    use_sex: bool
        Whether or not concatenate the last fully connected layer with a tabular feature that
        encodes sex. Also adds another fully connected layer.
    weights : str
        The pretrained weights for the model
    """
    return _agenet_3d(
        block=BuildingBlock_3D,
        blocks_per_stack=[3, 4, 6, 3],
        num_classes=num_classes,
        use_dropout=use_dropout,
        use_sex=use_sex,
        weights=weights,
    )
