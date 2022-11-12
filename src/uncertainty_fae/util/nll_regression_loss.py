import torch


def nll_regression_loss(input: torch.Tensor, target: torch.Tensor):
    """Negative Logliklihood for Regression Network with two outputs (mean and variance).

    TODO Docs
    """
    assert len(input.shape) == 2, 'Input Tensor in wrong Shape (probably batch dimension missing?)!'
    assert input.shape[1] == 2, 'Network outputs of wrong dimension!'

    mean = input[:, :1]
    variance = input[:, 1:]

    # Make Target a column if given as row
    if len(target.shape) == 1:
        target = target.unsqueeze(1)
    # Target shape "sanity check": for each batch exactly one target value
    assert target.shape == mean.shape

    batch_loss = torch.div(
        torch.add(
            torch.log(variance),
            torch.div(
                torch.square(torch.sub(target, mean)),
                variance
            )
        ),
        2
    )
    return torch.mean(batch_loss)
