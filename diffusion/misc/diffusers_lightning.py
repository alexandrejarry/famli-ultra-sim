import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from diffusers import DDPMScheduler, UNet2DModel
from datetime import datetime
import torchvision
from torchvision import transforms
import SimpleITK as sitk
import pandas as pd
import lightning as l
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import NeptuneLogger

device = torch.device("cuda")
torchvision.disable_beta_transforms_warning()

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
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1)  
])

class DiffusionModel(l.LightningModule):
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

    def forward(self, x, timesteps):
        return self.net(x, timesteps).sample

    def training_step(self, batch, batch_idx):
        x= batch  # Ignore labels if they exist
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
output_dir = f"outputs_lightning/output_{timestamp}/"
os.makedirs(output_dir, exist_ok=True)

parquet = "/mnt/raid/C1_ML_Analysis/CSV_files/extract_frames_Dataset_C_masked_resampled_256_spc075_wscores_meta_noflyto_1e-4.parquet"
df = pd.read_parquet(parquet, engine="pyarrow")
num_samples = 2000
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
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True, persistent_workers=False, prefetch_factor=2)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

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
trainer = l.Trainer(
    accelerator="gpu", 
    devices=4,  
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
trainer.fit(model, train_dataloader, val_dataloader)

final_model_path = os.path.join(output_dir, "model.pth")
torch.save(model.state_dict(), final_model_path)