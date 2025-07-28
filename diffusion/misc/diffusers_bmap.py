import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from diffusers import DDPMScheduler, UNet2DModel
from datetime import datetime
import torchvision
from torchvision import transforms
import numpy as np
import nrrd
import SimpleITK as sitk
import pandas as pd
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

device = torch.device("cuda")
torchvision.disable_beta_transforms_warning()

neptune_key = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkZWUyNGYzZi05ZDE0LTQwYjAtYTQzOS04M2QxZmQ5MTQ0MjcifQ=="

neptune_logger = NeptuneLogger(
    api_key=neptune_key,
    project="alexandrejarry/data-synthesis",
)

class DatasetFromDataFrame(Dataset):
    def __init__(self, root_dir, dataframe, transform=None):
        self.root_dir = root_dir
        self.df = dataframe
        self.transform = transform
        self.image_paths = []


        for i in range(self.df.shape[0]):
            img_name = self.df.iloc[i,0]
            self.image_paths.append(os.path.join(self.root_dir,img_name))
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
    
        if self.transform:
            image = self.transform(image)

        return image

transform = transforms.Compose([ 
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1)  
])

def generate_bmap(H, W, t, T, epsilon=0.03):
    gamma = 1.0 - (t / T) * epsilon
    gradient = np.linspace(1.0, gamma, H).reshape(H, 1)
    B_t = np.repeat(gradient, W, axis=1)
    return torch.tensor(B_t, dtype=torch.float32)


class DiffusionModel(pl.LightningModule):
    def __init__(self, lr=1e-5, num_train_timesteps=1000):
        super().__init__()
        self.save_hyperparameters()
        self.net = UNet2DModel(
            in_channels=1,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(256, 256, 512, 512, 1024, 1024),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D"
            ),
        )
        self.scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps, beta_schedule="linear")
        self.loss_fn = nn.MSELoss()

    def forward(self, x, timesteps):
        return self.net(x, timesteps).sample

    def training_step(self, batch, batch_idx):
        x = batch
        B, C, H, W = x.shape
        T = self.scheduler.config.num_train_timesteps
        timesteps = torch.randint(0, T, (B,), device=self.device).long()

        noise = torch.randn_like(x).to(self.device)
        noisy_images = torch.empty_like(x)

        for i in range(B):
            bmap = generate_bmap(H, W, timesteps[i].item(), T).to(x.device)
            bmap = bmap.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
            alpha_t = self.scheduler.alphas_cumprod[timesteps[i]].to(x.device)

            sqrt_alpha_B = torch.sqrt(alpha_t * bmap)
            sqrt_one_minus_alpha_B = torch.sqrt(1.0 - alpha_t * bmap)
            noisy_images[i] = sqrt_alpha_B * x[i] + sqrt_one_minus_alpha_B * noise[i]

        pred = self(noisy_images, timesteps)
        loss = self.loss_fn(pred, noise)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        B, C, H, W = x.shape
        T = self.scheduler.config.num_train_timesteps
        timesteps = torch.randint(0, T, (B,), device=self.device).long()

        noise = torch.randn_like(x).to(self.device)
        noisy_images = torch.empty_like(x)

        for i in range(B):
            bmap = generate_bmap(H, W, timesteps[i].item(), T).to(x.device)
            bmap = bmap.unsqueeze(0).unsqueeze(0)
            alpha_t = self.scheduler.alphas_cumprod[timesteps[i]].to(x.device)

            sqrt_alpha_B = torch.sqrt(alpha_t * bmap)
            sqrt_one_minus_alpha_B = torch.sqrt(1.0 - alpha_t * bmap)
            noisy_images[i] = sqrt_alpha_B * x[i] + sqrt_one_minus_alpha_B * noise[i]

        pred = self(noisy_images, timesteps)
        loss = self.loss_fn(pred, noise)

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.net.parameters(), lr=self.hparams.lr)

# Checkpoint directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"outputs_bmap/output_{timestamp}/"
os.makedirs(output_dir, exist_ok=True)

parquet = "/mnt/raid/C1_ML_Analysis/CSV_files/extract_frames_Dataset_C_masked_resampled_256_spc075_wscores_meta_noflyto_1e-4.parquet"
df = pd.read_parquet(parquet, engine="pyarrow")
num_samples = 50000
df_sampled = df.sample(n=num_samples, random_state=42).reset_index(drop=True)

train_root_dir = '/mnt/raid/C1_ML_Analysis'
full_dataset = DatasetFromDataFrame(root_dir=train_root_dir,dataframe=df_sampled, transform=transform)
train_ratio = 0.9
train_size = int(train_ratio * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
# Dataset & DataLoader
batch_size = 8
num_workers = 8
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True, persistent_workers=True, prefetch_factor=1)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=False, persistent_workers=True,)

# Define checkpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath=output_dir,  # Directory to save checkpoints
    filename="{epoch:02d}-{val_loss:.2f}",  # Naming format
    save_top_k=3,  # Save only the 3 best models
    monitor="val_loss",  # Monitor validation loss
    mode="min",  # Save models with the lowest val_loss
    save_last=True  # Always keep the last epoch checkpoint
)

# Early stopping to stop training when validation loss stops improving
early_stopping_callback = EarlyStopping(
    monitor="val_loss",
    patience=10,  # Stop if no improvement for 5 epochs
    mode="min",
    verbose=True
)

# PyTorch Lightning Trainer
trainer = pl.Trainer(
    accelerator="gpu", 
    logger=neptune_logger,
    devices=4,  
    strategy=DDPStrategy(find_unused_parameters=False),
    precision=32, 
    max_epochs=50,
    log_every_n_steps=100,
    enable_progress_bar=True,
    callbacks=[checkpoint_callback, early_stopping_callback],
)

# Train the model
model = DiffusionModel()
model = model.to(device)
model = model.float()
trainer.fit(model, train_dataloader, val_dataloader)

final_model_path = os.path.join(output_dir, "model.pth")
torch.save(model.state_dict(), final_model_path)