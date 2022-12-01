import torch
import torch.nn as nn
from torch import Tensor

from typing import Type, List, Optional


def conv1x1(channels_in: int, channels_out: int, stride: int = 1) -> nn.Conv2d:
    """1x1x1 convolution"""
    return nn.Conv2d(channels_in, channels_out, kernel_size=1, stride=stride, bias=False)


def conv1x1x1(channels_in: int, channels_out: int, stride: int = 1) -> nn.Conv3d:
    """1x1x1 convolution"""
    return nn.Conv3d(channels_in, channels_out, kernel_size=1, stride=stride, bias=False)


def conv3x3(channels_in: int, channels_out: int, stride: int = 1) -> nn.Conv2d:
    """3x3x3 convolution with padding"""
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
    """3x3x3 convolution with padding"""
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


def upconv2x2(channels_in: int, channels_out: int, stride: int = 1) -> nn.ConvTranspose2d:
    """2x2x2 up-convolution with padding"""
    return nn.ConvTranspose2d(
        channels_in, channels_out, kernel_size=2, stride=stride, padding=0, output_padding=0
    )


def upconv2x2x2(channels_in: int, channels_out: int, stride: int = 1) -> nn.ConvTranspose3d:
    """2x2x2 up-convolution with padding"""
    return nn.ConvTranspose3d(
        channels_in, channels_out, kernel_size=2, stride=stride, padding=0, output_padding=0
    )


class BuildingBlock_2D(nn.Module):
    """Building block for the 18-layer and 32-layer ResNet architecture"""

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
    """Building block for the 18-layer and 32-layer ResNet architecture"""

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


class BuildingBlockUp_2D(nn.Module):
    """Building block for the 18-layer and 32-layer ResNet custom autencoder architecture"""

    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        norm_layer=nn.BatchNorm2d,
        upsample1: Optional[nn.Module] = None,
        upsample2: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.upsample1 = upsample1
        self.conv1 = conv3x3(channels_in, channels_in)
        self.bn1 = norm_layer(channels_in)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(channels_in, channels_out)
        self.bn2 = norm_layer(channels_out)
        self.upsample2 = upsample2

    def forward(self, x: Tensor) -> Tensor:

        identity = x

        # Convolution 1
        if self.upsample1 is not None:
            out = self.upsample1(x)
        else:
            out = self.conv1(x)
            out = self.bn1(out)
        out = self.relu(out)

        # Convolution 2
        out = self.conv2(out)
        out = self.bn2(out)

        # Shortcut connection (identity or projection)
        if self.upsample2 is not None:
            identity = self.upsample2(x)

        out += identity
        out = self.relu(out)

        return out


class BuildingBlockUp_3D(nn.Module):
    """Building block for the 18-layer and 32-layer ResNet custom autencoder architecture"""

    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        norm_layer=nn.BatchNorm3d,
        upsample1: Optional[nn.Module] = None,
        upsample2: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.upsample1 = upsample1
        self.conv1 = conv3x3x3(channels_in, channels_in)
        self.bn1 = norm_layer(channels_in)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(channels_in, channels_out)
        self.bn2 = norm_layer(channels_out)
        self.upsample2 = upsample2

    def forward(self, x: Tensor) -> Tensor:

        identity = x

        # Convolution 1
        if self.upsample1 is not None:
            out = self.upsample1(x)
        else:
            out = self.conv1(x)
            out = self.bn1(out)
        out = self.relu(out)

        # Convolution 2
        out = self.conv2(out)
        out = self.bn2(out)

        # Shortcut connection (identity or projection)
        if self.upsample2 is not None:
            identity = self.upsample2(x)

        out += identity
        out = self.relu(out)

        return out


# --------------------------------------------------
# AgeNet14 3D Autoencoder
# --------------------------------------------------


class AgeNet_3D_Short_Autoencoder(nn.Module):
    def __init__(
        self,
        block: Type[BuildingBlock_3D],
        upblock: Type[BuildingBlockUp_3D],
        blocks_per_stack: List[int],
        use_sex: bool = False,
        sex_neurons: int = 16,
    ) -> None:
        super().__init__()

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

        self.upconv = nn.ConvTranspose3d(512, 256, kernel_size=3, stride=2)
        self.uplayer1 = self._stack_up_blocks(upblock, blocks_per_stack[2], 256, 128, 2, norm_layer)
        self.uplayer2 = self._stack_up_blocks(upblock, blocks_per_stack[1], 128, 64, 2, norm_layer)
        self.uplayer3 = self._stack_up_blocks(upblock, blocks_per_stack[0], 64, 64, 2, norm_layer)
        self.upsample = torch.nn.Upsample(scale_factor=2)
        self.conv3 = nn.Conv3d(64, 1, kernel_size=3, stride=1, padding=1)
        self.bn2 = norm_layer(1)
        self.conv4 = nn.Conv3d(1, 1, kernel_size=1, stride=1)

    def _stack_blocks(
        self,
        block: Type[BuildingBlock_3D],
        n_blocks: int,
        channels_in: int,
        channels_out: int,
        stride: int = 1,
        norm_layer=nn.BatchNorm3d,
    ) -> nn.Sequential:
        """Wrapper for stacking encoder blocks"""

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

    def _stack_up_blocks(
        self,
        block: Type[BuildingBlockUp_3D],
        n_blocks: int,
        channels_in: int,
        channels_out: int,
        scale_factor: int = 1,
        norm_layer=nn.BatchNorm3d,
    ) -> nn.Sequential:
        """Wrapper for stacking decoder blocks"""

        # Configure upsampling for the first block (= upsample1) in the stack and the skip connection (= upsample2)
        if scale_factor == 2:
            upsample1 = nn.Sequential(
                upconv2x2x2(channels_in, channels_in, scale_factor), norm_layer(channels_in)
            )
            upsample2 = nn.Sequential(
                upconv2x2x2(channels_in, channels_out, scale_factor), norm_layer(channels_out)
            )
        else:
            raise ValueError("Scale factor is expected to be 2, got {:d}".format(scale_factor))

        # Stack building blocks
        layers = []

        # First block
        layers.append(block(channels_in, channels_out, norm_layer, upsample1, upsample2))

        # Additional blocks (stride and downsample are no longer considered)
        for _ in range(1, n_blocks):
            layers.append(block(channels_out, channels_out))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)

        # Decoder
        x = self.upconv(x)
        x = self.uplayer1(x)
        x = self.uplayer2(x)
        x = self.uplayer3(x)

        x = self.upsample(x)
        x = self.conv3(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv4(x)

        return x


# --------------------------------------------------
# AgeNet18 3D Autoencoder
# --------------------------------------------------


class AgeNet_3D_Autoencoder(nn.Module):
    def __init__(
        self,
        block: Type[BuildingBlock_3D],
        upblock: Type[BuildingBlockUp_3D],
        blocks_per_stack: List[int],
        use_sex: bool = False,
        sex_neurons: int = 16,
    ) -> None:
        super().__init__()

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
        if self.use_sex:
            self.sex_layer = nn.Linear(1, self.sex_neurons)
            self.fc1 = nn.Linear(512, 512 - self.sex_neurons)
            self.fc_ = nn.Linear(512, 512)  # This layer does not exist in AgeNet18_3D
            self.unflatten = nn.Unflatten(1, (512, 1, 1, 1))
        self.upconv = nn.ConvTranspose3d(512, 512, 4)

        self.uplayer1 = self._stack_up_blocks(upblock, blocks_per_stack[3], 512, 256, 2, norm_layer)
        self.uplayer2 = self._stack_up_blocks(upblock, blocks_per_stack[2], 256, 128, 2, norm_layer)
        self.uplayer3 = self._stack_up_blocks(upblock, blocks_per_stack[1], 128, 64, 2, norm_layer)
        self.uplayer4 = self._stack_up_blocks(upblock, blocks_per_stack[0], 64, 64, 2, norm_layer)
        self.upsample = torch.nn.Upsample(scale_factor=1.75)
        self.conv2 = nn.Conv3d(64, 1, kernel_size=3, stride=1, padding=1)
        self.bn2 = norm_layer(1)
        self.conv3 = nn.Conv3d(1, 1, kernel_size=1, stride=1)

    def _stack_blocks(
        self,
        block: Type[BuildingBlock_3D],
        n_blocks: int,
        channels_in: int,
        channels_out: int,
        stride: int = 1,
        norm_layer=nn.BatchNorm3d,
    ) -> nn.Sequential:
        """Wrapper for stacking encoder blocks"""

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

    def _stack_up_blocks(
        self,
        block: Type[BuildingBlockUp_3D],
        n_blocks: int,
        channels_in: int,
        channels_out: int,
        scale_factor: int = 1,
        norm_layer=nn.BatchNorm3d,
    ) -> nn.Sequential:
        """Wrapper for stacking decoder blocks"""

        # Configure upsampling for the first block (= upsample1) in the stack and the skip connection (= upsample2)
        if scale_factor == 2:
            upsample1 = nn.Sequential(
                upconv2x2x2(channels_in, channels_in, scale_factor), norm_layer(channels_in)
            )
            upsample2 = nn.Sequential(
                upconv2x2x2(channels_in, channels_out, scale_factor), norm_layer(channels_out)
            )
        else:
            raise ValueError("Scale factor is expected to be 2, got {:d}".format(scale_factor))

        # Stack building blocks
        layers = []

        # First block
        layers.append(block(channels_in, channels_out, norm_layer, upsample1, upsample2))

        # Additional blocks (stride and downsample are no longer considered)
        for _ in range(1, n_blocks):
            layers.append(block(channels_out, channels_out))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        """
        # Bottleneck/Embedding
        x = self.avgpool(x)

        if self.use_sex:
            x = torch.flatten(x, 1)

            # Process embedding and shorten feature vector
            x = self.fc1(x)
            x = self.relu(x)
            # Introduce sex information by appending it to the embedding feature vector
            y = self.sex_layer(y)
            x = torch.cat((x,y), dim=1)
            # Fully incorporate sex information into embedding
            x = self.fc_(x)
            x = self.relu(x)

            x = self.unflatten(x)

        x = self.upconv(x)
        """

        # Decoder
        x = self.uplayer1(x)
        x = self.uplayer2(x)
        x = self.uplayer3(x)
        x = self.uplayer4(x)

        x = self.upsample(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)

        return x


# --------------------------------------------------
# AgeNet18 3D Light Autoencoder
# --------------------------------------------------


class AgeNet_3D_Light_Autoencoder(nn.Module):
    def __init__(
        self,
        block_3d: Type[BuildingBlock_3D],
        block_2d: Type[BuildingBlock_2D],
        upblock_3d: Type[BuildingBlockUp_3D],
        upblock_2d: Type[BuildingBlockUp_2D],
        blocks_per_stack: List[int],
        use_sex: bool = False,
        sex_neurons: int = 16,
    ) -> None:
        super().__init__()

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
        self.layer2 = self._stack_blocks_3d(block_3d, blocks_per_stack[1], 64, 128, 2, norm_layer)
        self.layer3 = self._stack_blocks_3d(block_3d, blocks_per_stack[2], 128, 256, 2, norm_layer)
        self.shrink = nn.Conv3d(256, 256, kernel_size=(1, 1, 7))
        self.layer4 = self._stack_blocks_2d(
            block_2d, blocks_per_stack[3], 256, 512, 2, self._norm_layer_2d
        )

        self.uplayer1 = self._stack_up_blocks_2d(
            upblock_2d, blocks_per_stack[3], 512, 256, 2, self._norm_layer_2d
        )
        self.expand = nn.ConvTranspose3d(256, 256, kernel_size=(1, 1, 8))
        self.uplayer2 = self._stack_up_blocks_3d(
            upblock_3d, blocks_per_stack[2], 256, 128, 2, norm_layer
        )
        self.uplayer3 = self._stack_up_blocks_3d(
            upblock_3d, blocks_per_stack[1], 128, 64, 2, norm_layer
        )
        self.uplayer4 = self._stack_up_blocks_3d(
            upblock_3d, blocks_per_stack[0], 64, 64, 2, norm_layer
        )
        self.upsample = torch.nn.Upsample(scale_factor=1.75)
        self.conv2 = nn.Conv3d(64, 1, kernel_size=3, stride=1, padding=1)
        self.bn2 = norm_layer(1)
        self.conv3 = nn.Conv3d(1, 1, kernel_size=1, stride=1)

    def _stack_blocks_3d(
        self,
        block: Type[BuildingBlock_3D],
        n_blocks: int,
        channels_in: int,
        channels_out: int,
        stride: int = 1,
        norm_layer=nn.BatchNorm3d,
    ) -> nn.Sequential:
        """Wrapper for stacking encoder blocks"""

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
        """Wrapper for stacking encoder blocks"""

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

    def _stack_up_blocks_3d(
        self,
        block: Type[BuildingBlockUp_3D],
        n_blocks: int,
        channels_in: int,
        channels_out: int,
        scale_factor: int = 1,
        norm_layer=nn.BatchNorm3d,
    ) -> nn.Sequential:
        """Wrapper for stacking decoder blocks"""

        # Configure upsampling for the first block (= upsample1) in the stack and the skip connection (= upsample2)
        if scale_factor == 2:
            upsample1 = nn.Sequential(
                upconv2x2x2(channels_in, channels_in, scale_factor), norm_layer(channels_in)
            )
            upsample2 = nn.Sequential(
                upconv2x2x2(channels_in, channels_out, scale_factor), norm_layer(channels_out)
            )
        else:
            raise ValueError("Scale factor is expected to be 2, got {:d}".format(scale_factor))

        # Stack building blocks
        layers = []

        # First block
        layers.append(block(channels_in, channels_out, norm_layer, upsample1, upsample2))

        # Additional blocks (stride and downsample are no longer considered)
        for _ in range(1, n_blocks):
            layers.append(block(channels_out, channels_out))

        return nn.Sequential(*layers)

    def _stack_up_blocks_2d(
        self,
        block: Type[BuildingBlockUp_2D],
        n_blocks: int,
        channels_in: int,
        channels_out: int,
        scale_factor: int = 1,
        norm_layer=nn.BatchNorm2d,
    ) -> nn.Sequential:
        """Wrapper for stacking decoder blocks"""

        # Configure upsampling for the first block (= upsample1) in the stack and the skip connection (= upsample2)
        if scale_factor == 2:
            upsample1 = nn.Sequential(
                upconv2x2(channels_in, channels_in, scale_factor), norm_layer(channels_in)
            )
            upsample2 = nn.Sequential(
                upconv2x2(channels_in, channels_out, scale_factor), norm_layer(channels_out)
            )
        else:
            raise ValueError("Scale factor is expected to be 2, got {:d}".format(scale_factor))

        # Stack building blocks
        layers = []

        # First block
        layers.append(block(channels_in, channels_out, norm_layer, upsample1, upsample2))

        # Additional blocks (stride and downsample are no longer considered)
        for _ in range(1, n_blocks):
            layers.append(block(channels_out, channels_out))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        # Encoder
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

        # Decoder
        x = self.uplayer1(x)
        x = torch.unsqueeze(x, 4)
        x = self.expand(x)
        x = self.uplayer2(x)
        x = self.uplayer3(x)
        x = self.uplayer4(x)

        x = self.upsample(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)

        return x


# --------------------------------------------------
# Wrapper functions
# --------------------------------------------------


def _agenet_3d_short_autoencoder(
    block: Type[BuildingBlock_3D],
    upblock: Type[BuildingBlockUp_3D],
    blocks_per_stack: List[int],
    use_sex: bool,
    weights: Optional[str],
) -> AgeNet_3D_Short_Autoencoder:

    model = AgeNet_3D_Short_Autoencoder(block, upblock, blocks_per_stack, use_sex)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict())

    return model


def _agenet_3d_autoencoder(
    block: Type[BuildingBlock_3D],
    upblock: Type[BuildingBlockUp_3D],
    blocks_per_stack: List[int],
    use_sex: bool,
    weights: Optional[str],
) -> AgeNet_3D_Autoencoder:

    model = AgeNet_3D_Autoencoder(block, upblock, blocks_per_stack, use_sex)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict())

    return model


def _agenet_3d_light_autoencoder(
    block_3d: Type[BuildingBlock_3D],
    block_2d: Type[BuildingBlock_2D],
    upblock_3d: Type[BuildingBlockUp_3D],
    upblock_2d: Type[BuildingBlockUp_2D],
    blocks_per_stack: List[int],
    use_sex: bool,
    weights: Optional[str],
) -> AgeNet_3D_Light_Autoencoder:

    model = AgeNet_3D_Light_Autoencoder(
        block_3d, block_2d, upblock_3d, upblock_2d, blocks_per_stack, use_sex
    )

    if weights is not None:
        model.load_state_dict(weights.get_state_dict())

    return model


def agenet14_3d_autoencoder(
    *, use_sex: bool = False, weights: Optional[str] = None
) -> AgeNet_3D_Short_Autoencoder:
    """
    Convolutional autoencoder network to pretraining AgeNet18-3D.

    The AgeNet18_3D model is adapted from the original ResNet18
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Parameters
    ----------
    blocks_per_stack : list
        Number of building blocks in each layer
    turn_on_sex: bool
        Whether or not concatenate the last fully connected layer with a tabular feature that
        encodes sex. Also adds another fully connected layer.
    weights : str
        The pretrained weights for the model
    """
    return _agenet_3d_short_autoencoder(
        block=BuildingBlock_3D,
        upblock=BuildingBlockUp_3D,
        blocks_per_stack=[2, 2, 2],
        use_sex=use_sex,
        weights=weights,
    )


def agenet18_3d_autoencoder(
    *, use_sex: bool = False, weights: Optional[str] = None
) -> AgeNet_3D_Autoencoder:
    """
    Convolutional autoencoder network to pretraining AgeNet18-3D.

    The AgeNet18_3D model is adapted from the original ResNet18
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Parameters
    ----------
    blocks_per_stack : list
        Number of building blocks in each layer
    turn_on_sex: bool
        Whether or not concatenate the last fully connected layer with a tabular feature that
        encodes sex. Also adds another fully connected layer.
    weights : str
        The pretrained weights for the model
    """
    return _agenet_3d_autoencoder(
        block=BuildingBlock_3D,
        upblock=BuildingBlockUp_3D,
        blocks_per_stack=[2, 2, 2, 2],
        use_sex=use_sex,
        weights=weights,
    )


def agenet18_3d_light_autoencoder(
    *, use_sex: bool = False, weights: Optional[str] = None
) -> AgeNet_3D_Light_Autoencoder:
    """
    Convolutional autoencoder network to pretraining AgeNet18-3D.

    The AgeNet18_3D model is adapted from the original ResNet18
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Parameters
    ----------
    blocks_per_stack : list
        Number of building blocks in each layer
    turn_on_sex: bool
        Whether or not concatenate the last fully connected layer with a tabular feature that
        encodes sex. Also adds another fully connected layer.
    weights : str
        The pretrained weights for the model
    """
    return _agenet_3d_light_autoencoder(
        block_3d=BuildingBlock_3D,
        block_2d=BuildingBlock_2D,
        upblock_3d=BuildingBlockUp_3D,
        upblock_2d=BuildingBlockUp_2D,
        blocks_per_stack=[2, 2, 2, 2],
        use_sex=use_sex,
        weights=weights,
    )


def agenet34_3d_autoencoder(
    *, use_sex: bool = False, weights: Optional[str] = None
) -> AgeNet_3D_Autoencoder:
    """
    Convolutional autoencoder network to pretraining AgeNet18-3D.

    The AgeNet18_3D model is adapted from the original ResNet18
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Parameters
    ----------
    blocks_per_stack : list
        Number of building blocks in each layer
    turn_on_sex: bool
        Whether or not concatenate the last fully connected layer with a tabular feature that
        encodes sex. Also adds another fully connected layer.
    weights : str
        The pretrained weights for the model
    """
    return _agenet_3d_autoencoder(
        block=BuildingBlock_3D,
        upblock=BuildingBlockUp_3D,
        blocks_per_stack=[3, 4, 6, 3],
        use_sex=use_sex,
        weights=weights,
    )
