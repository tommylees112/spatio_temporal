import pytorch_lightning as pl
from typing import Dict, Tuple, List
import torch
from torch import Tensor


class BaseModel(pl.LightningModule):
    """
    https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/master/pl_bolts/models/regression/linear_regression.py
    """

    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat["y_hat"], y)

        # logging for tensorboard
        tensorboard_logs = {"train_mse_loss": loss}
        progress_bar_metrics = tensorboard_logs
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return {
            "loss": loss,
            "log": tensorboard_logs,
            "progress_bar": progress_bar_metrics,
        }

    def configure_optimizers(self):
        return self.optimiser(self.parameters(), lr=self.hparams.learning_rate)

    #  Extra arguments for tensorboard information
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.loss(y_hat["y_hat"], y)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = self.loss(y_hat["y_hat"], y)
        return test_loss

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x for x in outputs]).mean()
        self.log("val_loss", val_loss)
        return

    def test_epoch_end(self, outputs):
        test_loss = torch.stack([x for x in outputs]).mean()
        self.log("test_loss", test_loss)
        return
