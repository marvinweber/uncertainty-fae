from typing import Tuple

import torch
from torch import Tensor, nn
from torchvision.models.inception import Inception3, InceptionOutputs


class RSNABoneageInceptionNetWithGender(nn.Module):
    def __init__(
        self, inception: Inception3, combination_layer_dropout=0.5, output_neurons=1
    ) -> None:
        super().__init__()

        self.inception = inception

        self.fc_gender = nn.Linear(1, 32)
        self.fc_combi_1 = nn.Linear(32 + inception.fc.out_features, 1000)
        self.dropout = nn.Dropout(p=combination_layer_dropout)
        self.fc_combi_2 = nn.Linear(1000, output_neurons)

    def forward(self, x: Tuple[Tensor, Tensor]):
        image, is_male = x
        if len(is_male.shape) == 1:
            is_male = torch.unsqueeze(is_male, 1)

        # Inception Net
        inception_output = self.inception.forward(image)
        if isinstance(inception_output, InceptionOutputs):
            inception_output = inception_output.logits

        # Gender Dense Net
        fc_gender_output = self.fc_gender(is_male)

        # Combination and Output
        fc_combi_input = torch.cat((inception_output, fc_gender_output), dim=1)
        x = self.fc_combi_1(fc_combi_input)
        x = self.dropout(x)
        output = self.fc_combi_2(x)
        return output
