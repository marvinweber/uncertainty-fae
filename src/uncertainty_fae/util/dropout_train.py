import torch


def dropout_train(m: torch.nn.Module) -> None:
    """Set all dropout layers in given module `m` to train mode."""
    if isinstance(m, torch.nn.modules.dropout.Dropout):
        m.train()
