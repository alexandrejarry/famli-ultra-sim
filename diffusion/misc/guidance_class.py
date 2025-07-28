import torch
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from generative.networks import nets
from generative.networks.schedulers import DDPMScheduler
from generative.inferers import DiffusionInferer
import neptune
import nrrd
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import os
from torchvision import transforms




class DDPMPL(pl.LightningModule):
    def __init__(self, guidance_path, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.guidance_path = guidance_path
        in_channels = 3  # 3 for input + 3 for guidance
        if hasattr(self.hparams, "in_channels"):
            in_channels = self.hparams.in_channels

        out_channels = 3  # Output remains the same
        if hasattr(self.hparams, "out_channels"):
            out_channels = self.hparams.out_channels

        self.model = nets.DiffusionModelUNet(
            spatial_dims=2,
            in_channels=in_channels,  # Increased to accept guidance image
            out_channels=out_channels,
            num_channels=(128, 256, 256),
            attention_levels=(False, True, True),
            num_res_blocks=1,
            num_head_channels=256,
        )

        self.scheduler = DDPMScheduler(num_train_timesteps=self.hparams.num_train_timesteps)
        self.inferer = DiffusionInferer(self.scheduler)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        return optimizer   

    def load_guidance_image(self, guidance_path):
        """Loads a single image and preprocesses it for all training samples."""
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        image = Image.open(guidance_path).convert("RGB")
        image = transform(image)
        return image  # Shape: (3, 256, 256)

    def training_step(self, train_batch, batch_idx):
        images, _ = train_batch  # Ignore labels since we don't use them
        guidance_image = self.load_guidance_image(self.guidance_path).to(self.device)

        # Create timesteps
        timesteps = torch.randint(
            0, self.inferer.scheduler.num_train_timesteps, (images.shape[0],), device=self.device
        ).long()

        # Concatenate guidance image with input
        noise = torch.rand_like(guidance_image)
        print("Noise shape before input:", noise.shape)

        noisy_image = torch.clamp(guidance_image + noise, 0, 1) 

        # Get model prediction
        noise_pred = self.inferer(
            inputs=images, diffusion_model=self.model, noise=noisy_image, timesteps=timesteps
        )

        loss = F.mse_loss(noise_pred.float(), noisy_image.float())

        self.log("train_loss", loss)

        return loss

    def validation_step(self, val_batch, batch_idx):
        images, _ = val_batch  # Ignore labels
        guidance_image = self.load_guidance_image(self.guidance_path).to(self.device)


        # Create timesteps
        timesteps = torch.randint(
            0, self.inferer.scheduler.num_train_timesteps, (images.shape[0],), device=self.device
        ).long()

        # Concatenate guidance image with input
        noise = torch.rand_like(guidance_image)
        noisy_image = torch.clamp(guidance_image + noise, 0, 1) 

        # Get model prediction
        noise_pred = self.inferer(
            inputs=images, diffusion_model=self.model, noise=noisy_image, timesteps=timesteps
        )

        loss = F.mse_loss(noise_pred.float(), noisy_image.float())

        self.log("val_loss", loss, sync_dist=True)

    def forward(self, x, timesteps=None, context=None):
        guidance_image = self.load_guidance_image(self.guidance_path).to(self.device)

        noise = torch.rand_like(guidance_image)
        noisy_image = torch.clamp(guidance_image + noise, 0, 1)  
        if timesteps is None:
            self.scheduler.set_timesteps(num_inference_steps=self.hparams.num_train_timesteps)
            return self.inferer.sample(
                input_noise=noisy_image,
                diffusion_model=self.model,
                scheduler=self.scheduler,
                verbose=False,
            )
        else:
            self.scheduler.set_timesteps(num_inference_steps=timesteps.item() if isinstance(timesteps, torch.Tensor) else timesteps)
            return self.inferer.sample(
                input_noise=noisy_image,
                diffusion_model=self.model,
                scheduler=self.scheduler,
                verbose=False,
            )

