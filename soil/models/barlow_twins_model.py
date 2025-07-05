import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from losses.barlow_loss import BarlowTwinsLoss
from models.projection_head import ProjectionHead
from functools import partial

def fn(warmup_steps, step):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    else:
        return 1.0


def linear_warmup_decay(warmup_steps):
    return partial(fn, warmup_steps)

class BarlowTwins(pl.LightningModule):
    def __init__(
        self,
        encoder,
        encoder_out_dim,
        num_training_samples,
        batch_size,
        lambda_coeff=5e-3,
        z_dim=128,
        learning_rate=1e-4,
        warmup_epochs=10,
        max_epochs=200,
    ):
        super().__init__()

        self.encoder = encoder
        self.projection_head = ProjectionHead(input_dim=encoder_out_dim, hidden_dim=encoder_out_dim, output_dim=z_dim)
        self.loss_fn = BarlowTwinsLoss(batch_size=batch_size, lambda_coeff=lambda_coeff, z_dim=z_dim)

        self.finetune_head = nn.Linear(encoder_out_dim, 1)
        
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        self.train_iters_per_epoch = num_training_samples // batch_size

    def forward(self, x):
        return self.encoder(x)

    # ---- SSL Step (dataloader_idx=0) ---- #
    def ssl_step(self, batch):
        x1, x2 = batch
        z1 = self.projection_head(self.encoder(x1))
        z2 = self.projection_head(self.encoder(x2))
        loss = self.loss_fn(z1, z2)
        self.log("ssl_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    # ---- Finetune Regression Step (dataloader_idx=1) ---- #
    # def finetune_step(self, batch):
    #     x, y = batch
    #     feats = self.encoder(x).detach()  # Freeze encoder, không backprop qua encoder
    #     preds = self.finetune_head(feats).squeeze()

    #     y = y.float().to(preds.device)
    #     loss = F.mse_loss(preds, y)

    #     rmse = torch.sqrt(loss)
    #     self.log("finetune_mse", loss, on_step=True, on_epoch=False, prog_bar=True)
    #     self.log("finetune_rmse", rmse, on_step=True, on_epoch=False, prog_bar=True)
    #     return loss

    def training_step(self, batch, batch_idx):
        return self.ssl_step(batch)
        # if batch_idx % 2 == 0:
            # return self.ssl_step(batch[0])
        # else:
            # return self.finetune_step(batch[1])
            
        # loss = self.shared_step(batch)
        # self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        # return loss

    def validation_step(self, batch, batch_idx):
        return 
    #     loss = self.shared_step(batch)
    #     self.log("val_loss", loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        warmup_steps = self.train_iters_per_epoch * self.warmup_epochs

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]