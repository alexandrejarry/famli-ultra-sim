import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import lightning as pl
from lightning.pytorch.strategies import DDPStrategy
from datetime import datetime
import json
from generative.losses.adversarial_loss import PatchAdversarialLoss
from generative.losses.perceptual import PerceptualLoss
from generative.networks import nets
from monai import transforms
from monai.data import DataLoader, Dataset
from monai.data.utils import partition_dataset
import os
import SimpleITK as sitk
from torch.nn.functional import one_hot
import pandas as pd
import numpy as np

device = torch.device("cuda")

class GaussianNoise(nn.Module):    
    def __init__(self, mean=0.0, std=0.05):
        super(GaussianNoise, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
    def forward(self, x):
        if self.training:
            return x + torch.normal(mean=self.mean, std=self.std, size=x.size(), device=x.device)
        return x
    
class RandCoarseShuffle(nn.Module):    
    def __init__(self, prob=0.75, holes=16, spatial_size=32):
        super(RandCoarseShuffle, self).__init__()
        self.t = transforms.RandCoarseShuffle(prob=prob, holes=holes, spatial_size=spatial_size)
    def forward(self, x):
        if self.training:
            return self.t(x)
        return x

class SaltAndPepper(nn.Module):    
    def __init__(self, prob=0.05):
        super(SaltAndPepper, self).__init__()
        self.prob = prob
    def __call__(self, x):
        noise_tensor = torch.rand(x.shape)
        salt = torch.max(x)
        pepper = torch.min(x)
        x[noise_tensor < self.prob/2] = salt
        x[noise_tensor > 1-self.prob/2] = pepper
        return x

class AutoEncoderKLPaired(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        latent_channels = 3
        if hasattr(self.hparams, "latent_channels"):
            latent_channels = self.hparams.latent_channels

        self.autoencoderkl = nets.AutoencoderKL(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_channels=(64, 128, 256, 512),
            latent_channels=latent_channels,
            num_res_blocks=2,
            attention_levels=(False, False, True, True),
            with_encoder_nonlocal_attn=False,
            with_decoder_nonlocal_attn=False,
        )
        self.perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="alex")

        self.automatic_optimization = False

        self.discriminator = nets.PatchDiscriminator(spatial_dims=2, num_layers_d=3, num_channels=64, in_channels=1, out_channels=1)        

        self.adversarial_loss = PatchAdversarialLoss(criterion="least_squares")        

        self.noise_transform = torch.nn.Sequential(
            GaussianNoise(0.0, 0.05),
            RandCoarseShuffle(),
            SaltAndPepper()     
        )
        
    def configure_optimizers(self):
        optimizer_g = optim.AdamW(self.autoencoderkl.parameters(),
                                lr=1e-4,
                                weight_decay=0.01)
        optimizer_d = optim.AdamW(self.discriminator.parameters(),
                                lr=1e-4,
                                weight_decay=0.01)
        return [optimizer_g, optimizer_d]

    def training_step(self, train_batch, batch_idx):
        x = train_batch[0]
        y = train_batch[1]

        optimizer_g, optimizer_d = self.optimizers()
        
        optimizer_g.zero_grad()

        reconstruction, z_mu, z_sigma = self.autoencoderkl(x)

        recons_loss = F.l1_loss(reconstruction.float(), y.float())
        p_loss = self.perceptual_loss(reconstruction.float(), y.float())
        kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        loss_g = recons_loss + (1e-6 * kl_loss) + (0.001 * p_loss)

        if self.trainer.current_epoch >= 0:
            logits_fake = self.discriminator(reconstruction.contiguous().float())[-1]
            generator_loss = self.adversarial_loss(logits_fake, target_is_real=True, for_discriminator=False)
            loss_g += 0.01 * generator_loss

        loss_g.backward()
        optimizer_g.step()
        
        loss_d = 0
        if self.trainer.current_epoch >= 0:
            
            optimizer_d.zero_grad()

            logits_fake = self.discriminator(reconstruction.contiguous().detach())[-1]
            loss_d_fake = self.adversarial_loss(logits_fake, target_is_real=False, for_discriminator=True)
            logits_real = self.discriminator(y.contiguous().detach())[-1]
            loss_d_real = self.adversarial_loss(logits_real, target_is_real=True, for_discriminator=True)
            discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

            loss_d = 0.01 * discriminator_loss

            loss_d.backward()
            optimizer_d.step()

        self.log("train_loss_g", loss_g)
        self.log("train_loss_d", loss_d)

        return {"train_loss_g": loss_g, "train_loss_d": loss_d} 

    def validation_step(self, val_batch, batch_idx):
        x = val_batch[0]
        y = val_batch[1]
        # with autocast(enabled=True):
        #     reconstruction, z_mu, z_sigma = self.autoencoderkl(x)
        #     recon_loss = F.l1_loss(x.float(), reconstruction.float())

        reconstruction, z_mu, z_sigma = self.autoencoderkl(x)
        recon_loss = F.l1_loss(y.float(), reconstruction.float())

        self.log("val_loss", recon_loss, sync_dist=True)

    def forward(self, images): 
        images = one_hot(images.long()).permute(0,3,1,2)   
        return self.autoencoderkl(images)

class SuperResDatasetFromCSV(Dataset):
    def __init__(self, csv_file, img_column, root_dir=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.img_column = img_column

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rel_path = self.df.iloc[idx][self.img_column]
        full_path = os.path.join(self.root_dir, rel_path) if self.root_dir else rel_path

        # Load image as numpy array
        image = sitk.GetArrayFromImage(sitk.ReadImage(full_path))  # shape: [1, H, W] or [H, W]
        if image.ndim == 3:
            image = image[0]  # keep only 2D if it’s a single-slice 3D image

        # Normalize to [0,1]
        image = image.astype(np.float32) / 255.0

        # Convert to torch tensor with shape [1, H, W]
        high_res = torch.from_numpy(image).unsqueeze(0)  # [1, 256, 256]

        # Downsample to 128x128 using bilinear interpolation
        low_res = torch.nn.functional.interpolate(
            high_res.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False
        ).squeeze(0)  # [1, 128, 128]

        return low_res, high_res
    
transform = transforms.Compose([
    # transforms.EnsureChannelFirst(channel_dim="1"),  # Ensures a single channel if input has none
    transforms.ToTensor()
])
train_data = SuperResDatasetFromCSV(csv_file='/mnt/raid/C1_ML_Analysis/CSV_files/ALL_C2_cines_gt_ga_withmeta_20221031_butterfly_train.csv', img_column='file_path', root_dir='/mnt/raid/C1_ML_Analysis/')
val_data = SuperResDatasetFromCSV(csv_file='/mnt/raid/C1_ML_Analysis/CSV_files/ALL_C2_cines_gt_ga_withmeta_20221031_butterfly_valid.csv', img_column='file_path', root_dir='/mnt/raid/C1_ML_Analysis/')

train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4, persistent_workers=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False, num_workers=4, persistent_workers=True)

model = AutoEncoderKLPaired()
model = model.to(device)

trainer = pl.Trainer(
    accelerator="gpu", 
    devices=4,  
    strategy=DDPStrategy(find_unused_parameters=True),
    precision=16, 
    max_epochs=50,
    log_every_n_steps=10,
    enable_progress_bar=True,
)

trainer.fit(model, train_loader, val_loader)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"autoencoder/output_{timestamp}/"
os.makedirs(output_dir, exist_ok=True)
final_model_path = os.path.join(output_dir, "model.pth")
torch.save(model.state_dict(), final_model_path)