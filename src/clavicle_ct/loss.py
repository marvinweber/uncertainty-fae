import torch
from torch.nn.modules.loss import _Loss


class AgeLoss_1(_Loss):
    """f(x,y) = 2*(x-y)^2"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        a = torch.pow(x - y, 2)

        loss = torch.mean(torch.mul(2, a))

        return loss


class AgeLoss_2(_Loss):
    """f(x,y) = (x-y)^2 + 3*(|x-y|)^3"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        a = torch.pow(x - y, 2)
        b = torch.pow(torch.abs(x - y), 3)

        loss = torch.mean(torch.add(a, torch.mul(3, b)))

        return loss


class AgeLoss_2_Unscaled(_Loss):
    """f(x,y) = (x-y)^2 + (|x-y|)^3"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        a = torch.pow(x - y, 2)
        b = torch.pow(torch.abs(x - y), 3)

        loss = torch.mean(torch.add(a, b))

        return loss


class AgeLoss_3(_Loss):
    """f(x,y) = (x-y)^2 + 0.03*exp^(8*|x-y|) - 0.03"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        a = torch.pow(x - y, 2)  # (x-y)^2
        b1 = 0.03
        b2 = torch.exp(torch.mul(8, torch.abs(x - y)))  # exp^(8*|x-y|)
        b = torch.mul(b1, b2)
        c = -0.03

        loss = torch.mean(torch.add(a, torch.add(b, c)))

        return loss
