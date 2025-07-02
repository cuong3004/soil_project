import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from datasets.soil_dataset_ssl import SoilDatasetSSL
from datasets.soil_dataset_finetune import SoilDatasetFinetune
from transforms.barlow_train_transform import BarlowTwinsTrainTransform
from transforms.barlow_val_transform import BarlowTwinsValTransform
from utils.normalization import normalization
from models.encoder import get_encoder
from models.barlow_twins_model import BarlowTwins
from callbacks.online_finetuner import OnlineFineTuner

# ==== Config ==== #
DATA_ROOT_SSL = "/content/data_soid_image"
DATA_ROOT_FINE = "/content/data_folder"
BATCH_SIZE = 256
NUM_WORKERS = 64
MAX_EPOCHS = 200
Z_DIM = 512
ENCODER_OUT_DIM = 576  #
LEARNING_RATE = 1e-4
WARMUP_EPOCHS = 10



# ==== Transforms ==== #
train_transform = BarlowTwinsTrainTransform(
    input_height=224,
    gaussian_blur=False,
    jitter_strength=0.5,
    normalize=normalization()
)

val_transform = BarlowTwinsValTransform(
    input_height=224,
    normalize=normalization()
)

# ==== Datasets ==== #
ssl_dataset = SoilDatasetSSL(root_dir=DATA_ROOT_SSL, transform=train_transform)
finetune_dataset = SoilDatasetFinetune(root_dir=DATA_ROOT_FINE, transform=val_transform)

# ==== DataLoaders ==== #
ssl_loader = DataLoader(
    ssl_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    drop_last=True
)

finetune_loader = DataLoader(
    finetune_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    drop_last=True
)

class SoilDataModule(pl.LightningDataModule):
    def __init__(self, ssl_loader, finetune_loader):
        super().__init__()
        self.ssl_loader = ssl_loader
        self.finetune_loader = finetune_loader

    def train_dataloader(self):
        return [self.ssl_loader, self.finetune_loader]

data_module = SoilDataModule(
    ssl_loader,
    finetune_loader
)

# ==== Model ==== #
encoder = get_encoder()

model = BarlowTwins(
    encoder=encoder,
    encoder_out_dim=ENCODER_OUT_DIM,
    num_training_samples=len(ssl_dataset),
    batch_size=BATCH_SIZE,
    lambda_coeff=5e-3,
    z_dim=Z_DIM,
    learning_rate=LEARNING_RATE,
    warmup_epochs=WARMUP_EPOCHS,
    max_epochs=MAX_EPOCHS,
)

# ==== Callback ==== #
online_finetuner = OnlineFineTuner(encoder_output_dim=ENCODER_OUT_DIM, num_classes=10)
checkpoint_callback = ModelCheckpoint(every_n_epochs=100, save_top_k=-1, save_last=True)

# ==== Trainer ==== #
trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator="auto",
    devices=1,
    callbacks=[online_finetuner, checkpoint_callback],
)

# ==== Train ==== #
trainer.fit(model, datamodule=data_module)