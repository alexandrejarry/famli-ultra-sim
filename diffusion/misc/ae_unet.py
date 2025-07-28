import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from datetime import datetime
import json
from generative.losses.adversarial_loss import PatchAdversarialLoss
from generative.losses.perceptual import PerceptualLoss
from generative.networks import nets
from monai import transforms
from monai.data import DataLoader, Dataset
from monai.data.utils import partition_dataset
from monai.networks.nets import UNet
import os
import SimpleITK as sitk

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

        self.autoencoderkl = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(64, 128, 256, 512),
            num_res_units=2,
            strides = (2, 2, 2)
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

        reconstruction = self.autoencoderkl(x)

        recons_loss = F.l1_loss(reconstruction.float(), y.float())
        p_loss = self.perceptual_loss(reconstruction.float(), y.float())
        loss_g = recons_loss  + (0.001 * p_loss)

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

        reconstruction = self.autoencoderkl(x)
        recon_loss = F.l1_loss(y.float(), reconstruction.float())

        self.log("val_loss", recon_loss, sync_dist=True)


    def forward(self, images): 
        return self.autoencoderkl(images)

class CustomDataset(Dataset):
    def __init__(self, g_dict, d_dict, transform=None):
        print("loading data")
        self.transform = transform
        self.g_images = []
        self.g_dict = g_dict
        self.d_images = []
        self.d_dict = d_dict

        for g_path in self.g_dict:
            for i in self.g_dict[g_path]:
                g_file_path = os.path.join(g_path, os.listdir(g_path)[i])
                g_data = sitk.GetArrayFromImage(sitk.ReadImage(g_file_path))
                self.g_images.append(g_data)

        for d_path in self.d_dict:
            for i in self.d_dict[d_path]:
                d_file_path = os.path.join(d_path, os.listdir(d_path)[i])
                d_data = sitk.GetArrayFromImage(sitk.ReadImage(d_file_path))
                self.d_images.append(d_data)
        print("data loading complete")

    def __len__(self):
        return len(self.g_images)
    
    def __getitem__(self, idx):
        g_image = self.g_images[idx]
        d_image = self.d_images[idx]
        if self.transform:
            g_image = self.transform(g_image).unsqueeze(0)
            d_image = self.transform(d_image).unsqueeze(0)

        

        return (d_image,g_image)
    

transform = transforms.Compose([
    transforms.EnsureChannelFirst(channel_dim="1"), 
    transforms.ToTensor()
])

with open("dictionary_diffusor.json","r") as f:
    dictionary_diffusor = json.load(f)

with open("dictionary_guided.json","r") as f:
    dictionary_guided = json.load(f)

batch_size = 32

full_dataset = CustomDataset(g_dict=dictionary_guided, d_dict=dictionary_diffusor ,transform=transform)
train_data, val_data = partition_dataset(full_dataset, ratios=[0.8, 0.2], shuffle=True, seed=42)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)

model = AutoEncoderKLPaired()

trainer = pl.Trainer(
    accelerator="gpu", 
    devices=4,  
    strategy=DDPStrategy(find_unused_parameters=True),
    precision=32, 
    max_epochs=200,
    log_every_n_steps=10,
    enable_progress_bar=True,
)

trainer.fit(model, train_loader, val_loader)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"autoencoder/output_{timestamp}/"
os.makedirs(output_dir, exist_ok=True)
final_model_path = os.path.join(output_dir, "model.pth")
torch.save(model.state_dict(), final_model_path)