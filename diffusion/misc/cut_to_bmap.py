import torch
import pytorch_lightning as pl
from diffusers import DDPMScheduler, UNet2DModel
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import SimpleITK as sitk
import plotly.express as px
import os

device = torch.device("cuda")

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
    
checkpoint_path = "/mnt/raid/home/ajarry/data/outputs_bmap/output_20250429_100112/model.pth"
model = DiffusionModel()
state_dict = torch.load(checkpoint_path)
model.load_state_dict(state_dict, strict=False)
model = model.to(device)

def guidance_loss(image, target):
    return torch.abs(image - target).mean()

scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")
# Set up sampling parameters
image_size = (1, 1, 256, 256)  # Adjust shape (batch, channels, height, width)
num_inference_steps = 50  # Can be reduced for faster generation
scheduler.set_timesteps(num_inference_steps)
# Start with pure noise
noisy_image = torch.randn(image_size).to(device)
# Sample timesteps
timesteps = scheduler.timesteps.to(device)
guidance_loss_scale = 200 # Explore changing this to 5, or 100
totensor = transforms.ToTensor()

def generate_bmap(shape, device=device):
    b, c, h, w = shape
    linear_map = torch.linspace(1.0, 0.0, steps=h, device=device).view(1, 1, h, 1)
    bmap = linear_map.expand(b, c, h, w)
    return bmap

parent_dir = "/mnt/raid/C1_ML_Analysis/simulated_data_export/placenta_simu/IB1_label11"
# Loop through all subdirectories

for file in os.listdir(parent_dir):
    if "C4_cut.nrrd" in file:
        file_path = os.path.join(parent_dir,file)
        save_path = file_path.replace("cut","bmap_guided")

        data = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
        images = []
        for i in range(data.shape[0]):
            images.append(totensor(data[i]))

        guided_frames = []
        cpt = 0
        for im in images:
            cpt += 1
            print("Frame: ", cpt)
            x = torch.randn(1, 1, 256, 256).to(device)
            target_image = im.to(device)

            for i, t in tqdm(enumerate(scheduler.timesteps), ):
                
                with torch.no_grad():
                    noise_pred = model(x, t)

                # Set x.requires_grad to True
                x = x.detach().requires_grad_()

                # Get the predicted x0
                x0 = scheduler.step(noise_pred, t, x).pred_original_sample

                # Get B-map
                bmap = generate_bmap(x.shape, device)

                # Apply B-map modulation
                x0 = bmap * x0 + (1.0 - bmap) * x0  # guided update

                # Calculate loss
                loss = guidance_loss(x0, target_image) * guidance_loss_scale
                if i % 10 == 0:
                    print(i, "loss:", loss.item())

                # Get gradient
                cond_grad = -torch.autograd.grad(loss, x)[0]

                # Modify x based on this gradient
                x = x.detach() + cond_grad
    
                x = scheduler.step(noise_pred, t, x).prev_sample

            guided_frames.append(x)
            
        guided_frames_cpu = [frame.cpu().numpy() for frame in guided_frames]
        video = np.stack(guided_frames_cpu, axis=0).squeeze(1).squeeze(1)
        img = sitk.GetImageFromArray(video)
        print("Saved : ", save_path)
        sitk.WriteImage(img, save_path)