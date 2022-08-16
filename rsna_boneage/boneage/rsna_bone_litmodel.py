import torch.optim
from pytorch_lightning.core.lightning import LightningModule
from torch import nn, squeeze, vstack
from torchvision.models.inception import InceptionOutputs

from .rsna_boneage_inception import RSNABoneageInceptionNetWithGender
from .rsna_boneage_resnet import RSNABoneageResNetWithGender


class RSNABoneageLitModel(LightningModule):
    def __init__(self, net: nn.Module, lr: float = 3e-4,
                 mc_iterations=100, weight_decay_adam: float = 0,
                 undo_boneage_rescaling=False) -> None:
        super().__init__()

        self.net = net
        self.lr = lr
        self.weight_decay_adam = weight_decay_adam

        self.mae = nn.L1Loss()
        self.mse = nn.MSELoss()

        self.mc_iterations = mc_iterations
        self.undo_boneage_rescaling = undo_boneage_rescaling

        self.save_hyperparameters(ignore=['net'])

    def forward(self, x):
        logits = self.net.forward(x)

        # Workaround to make this LitModel usable with Inception, too.
        if isinstance(logits, InceptionOutputs):
            logits = logits.logits

        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits = self.forward(x)
        logits = squeeze(logits, dim=1)

        mae = self.mae(logits, y)
        self.log('mae', mae)

        loss = self.mse(logits, y)
        self.log('loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits = self.forward(x)
        logits = squeeze(logits, dim=1)

        mae = self.mae(logits, y)
        self.log('val_mae', mae)

        loss = self.mse(logits, y)
        self.log('val_loss', loss)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        # Forward pass
        logits = self.forward(x)
        logits = squeeze(logits, dim=1)

        mae = self.mae(logits, y)
        self.log('test_mae', mae)

        loss = self.mse(logits, y)
        self.log('test_loss', loss)

    def forward_with_mc(self, batch):
        # Enable Dropout Layers in Network for MC
        self.net.dropout.train()
        # Workarround for Gender Nets -> TODO: Abstraction into Interface
        if isinstance(self.net, RSNABoneageInceptionNetWithGender):
            self.net.inception.dropout.train()
        if isinstance(self.net, RSNABoneageResNetWithGender):
            self.net.resnet.dropout.train()

        preds = [self.forward(batch).cpu() for _ in range(self.mc_iterations)]
        if self.undo_boneage_rescaling:
            preds = [self._undo_rescale_boneage(pred) for pred in preds]
        preds = vstack(preds)

        quantile_5 = preds.quantile(0.05)
        quantile_17 = preds.quantile(0.17)
        quantile_83 = preds.quantile(0.83)
        quantile_95 = preds.quantile(0.95)
        metrics = {
            'mean': preds.mean(),
            'median': preds.median(),
            'std': preds.std(),
            'var': preds.var(),
            'quantile_5': quantile_5,
            'quantile_17': quantile_17,
            'quantile_83': quantile_83,
            'quantile_95': quantile_95,
            'uncertainty': preds.std(),
            'uncertainty_17_83_range': quantile_83 - quantile_17,
            'uncertainty_5_95_range': quantile_95 - quantile_5,
            'predictions': preds,
        }
        return metrics['mean'], metrics

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay_adam)

    def _undo_rescale_boneage(self, boneage_rescaled: int):
        lower_bound = 0
        upper_bound = 230
        return (boneage_rescaled * (upper_bound - lower_bound)) + lower_bound
