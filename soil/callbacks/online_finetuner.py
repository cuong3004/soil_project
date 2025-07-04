# from functools import partial
from typing import Sequence, Tuple, Union

# import matplotlib.pyplot as plt
# import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision.transforms as transforms
# import torchvision.transforms.functional as VisionF
from pytorch_lightning.callbacks import Callback
from torch import Tensor
# from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy
# from torchvision.datasets import CIFAR10
# from torchvision.models.resnet import resnet18
# from torchvision.utils import make_grid


class OnlineFineTuner(Callback):
    def __init__(
        self,
        encoder_output_dim: int,
        num_classes: int,
    ) -> None:
        super().__init__()

        self.optimizer: torch.optim.Optimizer

        self.encoder_output_dim = encoder_output_dim
        self.num_classes = num_classes

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # add linear_eval layer and optimizer
        pl_module.online_finetuner = nn.Linear(self.encoder_output_dim, 1).to(pl_module.device)
        self.optimizer = torch.optim.Adam(pl_module.online_finetuner.parameters(), lr=1e-4)

    def extract_online_finetuning_view(
        self, batch: Sequence, device: Union[str, torch.device]
    ) -> Tuple[Tensor, Tensor]:
        finetune_view, y = batch
        finetune_view = finetune_view.to(device)
        y = y.to(device)

        return finetune_view, y

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
    ) -> None:
        x, y = self.extract_online_finetuning_view(batch, pl_module.device)

        with torch.no_grad():
            feats = pl_module(x)

        feats = feats.detach()
        with torch.enable_grad():
            preds = pl_module.online_finetuner(feats)
            # print(preds, y)
            # print(preds.shape, y.shape)
            loss = F.mse_loss(preds.squeeze(), y)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        rmse = torch.sqrt(loss)
        # pl_module.log("online_train_rmse", rmse, on_step=False, on_epoch=True)
        # rmse = torch.sqrt(loss)
        self.log("finetune_mse", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("finetune_rmse", rmse, on_step=False, on_epoch=True, prog_bar=True)
        # pl_module.log("online_train_loss", loss, on_step=True, on_epoch=False)
        # acc = accurac
        # pl_module.log("online_train_loss", loss, on_step=True, on_epoch=False)
        # acc = accuracy(F.softmax(preds, dim=1), y, task="multiclass", num_classes=10)
        # pl_module.log("online_train_acc", acc, on_step=True, on_epoch=False)
        # pl_module.log("online_train_loss", loss, on_step=True, on_epoch=False)
