from typing import Optional

import torch
import torch.optim
from pytorch_lightning.core.module import LightningModule
from torch import nn


class LitClavicleAutoencoder(LightningModule):
    def __init__(self, net, n_inputs: int = 1, lr: float = 3e-4) -> None:
        super().__init__()

        self.net = net
        self.n_inputs = n_inputs
        self.lr = lr
        self.loss = nn.MSELoss()

        self.save_hyperparameters(ignore=["net"])

    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor] = None):
        if self.n_inputs == 1:
            return self.net(x)
        elif self.n_inputs == 2:
            return self.net(x, z)

    def training_step(self, batch, batch_idx):
        # Forward pass
        y = torch.empty([])
        logits = torch.empty([])
        if self.n_inputs == 1:
            x, y = batch
            logits = self.forward(x)
        elif self.n_inputs == 2:
            x, y, z = batch
            logits = self.forward(x, z)

        # Calculate loss
        loss = self.loss(logits, y)
        self.log("loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        # Forward pass
        y = torch.empty([])
        logits = torch.empty([])
        if self.n_inputs == 1:
            x, y = batch
            logits = self.forward(x)
        elif self.n_inputs == 2:
            x, y, z = batch
            logits = self.forward(x, z)
        else:
            raise ValueError("Bad number of inputs.")

        # Calculate loss
        loss = self.loss(logits, y)
        self.log("val_loss", loss)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
