from typing import Any, Callable, List, Optional, Tuple, Type, Union

import torch
from torch import Tensor, nn
from torchvision.models.resnet import (BasicBlock, Bottleneck, ResNet, load_state_dict_from_url,
                                       model_urls)


class RSNABoneageResNet(ResNet):

    def __init__(self, block: Type[Union[BasicBlock, Bottleneck]], layers: List[int],
                 num_classes: int = 1000, zero_init_residual: bool = False, groups: int = 1,
                 width_per_group: int = 64,
                 replace_stride_with_dilation: Optional[List[bool]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__(block, layers, num_classes, zero_init_residual,
                         groups, width_per_group, replace_stride_with_dilation, norm_layer)
        self.dropout = nn.Dropout(0.5)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


class RSNABoneageResNetWithGender(nn.Module):

    def __init__(self, resnet: RSNABoneageResNet, combination_layer_dropout=0.5,
                 output_neurons=1) -> None:
        super().__init__()

        self.resnet = resnet

        self.fc_gender = nn.Linear(1, 32)
        self.fc_combi_1 = nn.Linear(32 + resnet.fc.out_features, 1000)
        self.dropout = nn.Dropout(p=combination_layer_dropout)
        self.fc_combi_2 = nn.Linear(1000, output_neurons)

    def forward(self, x: Tuple[Tensor, Tensor]):
        image, is_male = x
        if len(is_male.shape) == 1:
            is_male = torch.unsqueeze(is_male, 1)

        # ResNet
        resnet_output = self.resnet.forward(image)

        # Gender Dense Net
        fc_gender_output = self.fc_gender(is_male)

        # Combination and Output
        fc_combi_input = torch.cat((resnet_output, fc_gender_output), dim=1)
        x = self.fc_combi_1(fc_combi_input)
        x = self.dropout(x)
        output = self.fc_combi_2(x)
        return output


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    model = RSNABoneageResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)
