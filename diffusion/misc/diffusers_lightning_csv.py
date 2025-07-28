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
import lightning as L
from lightning.pytorch.core import LightningModule, LightningDataModule

from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy

from lightning.pytorch.loggers import NeptuneLogger
import torch.nn.functional as F

device = torch.device("cuda")
torchvision.disable_beta_transforms_warning()

class Resize2D:
    def __init__(self, target_size, mode='nearest'):
        """
        Resize a 2D tensor [B, C, H, W] to the target size.

        Args:
            target_size (tuple): (H, W), use -1 to keep that dimension unchanged
        """
        assert len(target_size) == 2, "target_size must be a tuple of (H, W)"
        self.target_size = target_size
        self.mode = mode

    def __call__(self, x):
        """
        Resize the input tensor.

        Args:
            x (torch.Tensor): [B, C, H, W]

        Returns:
            torch.Tensor: resized tensor [B, C, H', W']
        """
        assert x.ndim == 4, "Input tensor must have shape [B, C, H, W]"
        B, C, H, W = x.shape

        out_H = self.target_size[0] if self.target_size[0] != -1 else H
        out_W = self.target_size[1] if self.target_size[1] != -1 else W

        resized = F.interpolate(x, size=(out_H, out_W), mode=self.mode)
        return resized
    
class USButterflyBlindSweep(Dataset):
    def __init__(self, df, mount_point = "./", img_column='img_path', transform=None, num_frames=-1, continous_frames=False):
        self.df = df
        self.mount_point = mount_point
        self.transform = transform
        self.img_column = img_column
        self.keys = self.df.index
        self.num_frames = num_frames
        self.continous_frames = continous_frames 

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.mount_point, self.df.iloc[idx][self.img_column])

        try:
            img = sitk.ReadImage(img_path)
            img_np = sitk.GetArrayFromImage(img)
            img_t = torch.tensor(img_np, dtype=torch.float32)
            
            if img.GetNumberOfComponentsPerPixel() == 1:
                img_t = img_t.unsqueeze(-1)
            elif  img.GetNumberOfComponentsPerPixel() == 3:
                img_t = img_t[:,:,:,0].unsqueeze(-1)

            
            img_t = img_t.permute(3, 0, 1, 2)/255.0  # Change to (C, D, H, W)             

            if self.num_frames > 0:
                
                idx = torch.randint(low=0, high=img_t.shape[1], size=(self.num_frames,))
                idx = idx.sort().values

            img_t = img_t[:, idx, :, :]
                    
        except:
            print("Error reading cine: " + img_path)
            n = self.num_frames if self.num_frames > 0 else 1
            img_t = torch.zeros(1, n, 256, 256, dtype=torch.float32)

        if self.transform:
            img_t = self.transform(img_t)

        return img_t.permute(1,0,2,3)
    
class USButterflyBlindSweepDataModule(LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.df_train = pd.read_csv(self.hparams.csv_train)
        self.df_val = pd.read_csv(self.hparams.csv_valid)

        self.train_transform = None
        self.valid_transform = None

    @staticmethod
    def add_data_specific_args(parent_parser):

        group = parent_parser.add_argument_group("USButterflyBlindSweepDataModule")
        group.add_argument('--batch_size', type=int, default=2)
        group.add_argument('--num_workers', type=int, default=6)
        group.add_argument('--num_frames', type=int, default=8)
        group.add_argument('--img_column', type=str, default="img")
        group.add_argument('--csv_train', type=str, default=None, required=True)
        group.add_argument('--csv_valid', type=str, default=None, required=True)
        group.add_argument('--mount_point', type=str, default="./")
        group.add_argument('--drop_last', type=int, default=0)

        return parent_parser
        
    
    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        self.train_ds = USButterflyBlindSweep(self.df_train, self.hparams.mount_point, img_column=self.hparams.img_column, transform=self.train_transform, num_frames=self.hparams.num_frames, continous_frames=True)
        self.val_ds = USButterflyBlindSweep(self.df_val, self.hparams.mount_point, img_column=self.hparams.img_column, transform=self.valid_transform, num_frames=self.hparams.num_frames, continous_frames=True)
    
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, persistent_workers=True, pin_memory=True, drop_last=bool(self.hparams.drop_last), shuffle=True, prefetch_factor=2,collate_fn=self.collate_fn)
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, drop_last=bool(self.hparams.drop_last), prefetch_factor=2,collate_fn=self.collate_fn)
    def collate_fn(self,batch):
        return torch.cat(batch, dim=0)

class DiffusionModel(LightningModule):
    def __init__(self, lr=1e-5, num_train_timesteps=1000):
        super().__init__()
        self.save_hyperparameters()
        self.net = UNet2DModel(
            in_channels=1,  # the number of input channels, 3 for RGB images
            out_channels=1,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(256, 256, 512, 512, 1024, 1024),  # the number of output channes for each UNet block
            down_block_types=( 
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D", 
                "DownBlock2D", 
                "DownBlock2D", 
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "DownBlock2D",
            ), 
            up_block_types=(
                "UpBlock2D",  # a regular ResNet upsampling block
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D", 
                "UpBlock2D", 
                "UpBlock2D", 
                "UpBlock2D"  
            ),
        )
        self.scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps, beta_schedule="linear")
        self.loss_fn = nn.MSELoss()
        self.resize = Resize2D((128,128))

    def forward(self, x, timesteps):
        return self.net(x, timesteps).sample

    def training_step(self, batch, batch_idx):
        x= self.resize(batch)  # Ignore labels if they exist
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (x.size(0),), device=self.device).long()
        noise = torch.randn_like(x).to(self.device)
        noisy_images = self.scheduler.add_noise(x, noise, timesteps)

        pred = self(noisy_images, timesteps)  # Model forward pass
        loss = self.loss_fn(pred, noise)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (x.size(0),), device=self.device)
        outputs = self(x, timesteps)
        loss = self.loss_fn(outputs, x)

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.net.parameters(), lr=self.hparams.lr)
        return optimizer

# Checkpoint directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"outputs_lightning/CSVoutput_{timestamp}/"
os.makedirs(output_dir, exist_ok=True)

train_root_dir = '/mnt/raid/C1_ML_Analysis/CSV_files/ALL_C2_cines_gt_ga_withmeta_20221031_butterfly_train.csv'
val_root_dir = '/mnt/raid/C1_ML_Analysis/CSV_files/ALL_C2_cines_gt_ga_withmeta_20221031_butterfly_valid.csv' 

# Dataset & DataLoader
batch_size = 8
num_workers = 16
params = {'batch_size':batch_size, 'num_workers':num_workers, 'num_frames':8, 'img_column':'file_path', 'csv_train':train_root_dir, 'csv_valid':val_root_dir, 'mount_point':'/mnt/raid/C1_ML_Analysis', 'drop_last':0}
dm = USButterflyBlindSweepDataModule(**params)

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
trainer = Trainer(
    accelerator="gpu", 
    devices=torch.cuda.device_count(),  
    strategy=DDPStrategy(find_unused_parameters=False),
    precision=16, 
    max_epochs=50,
    log_every_n_steps=10,
    enable_progress_bar=True,
    callbacks=[checkpoint_callback, early_stopping_callback],
)

# Train the model
model = DiffusionModel()
model = model.to(device)
model = model.float()
trainer.fit(model, dm)

final_model_path = os.path.join(output_dir, "model.pth")
torch.save(model.state_dict(), final_model_path)