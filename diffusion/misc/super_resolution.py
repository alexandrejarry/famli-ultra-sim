from diffusers.models.autoencoders import AutoencoderKL
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import numpy as np
import SimpleITK as sitk

device = torch.device("cuda")

class PairedResDataset(Dataset):
    def __init__(self, csv_paths, transform=None):
        self.paths = open(csv_paths).read().splitlines()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)  # shape: HxW or DxHxW
        img = np.squeeze(img)  # Drop extra dims if needed

        # Normalize and convert to 3-channel if needed
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)
        img = np.stack([img] * 3, axis=0)  # 3xHxW

        highres = torch.tensor(img).float()
        lowres = F.interpolate(highres.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False).squeeze(0)

        return lowres, highres

model = AutoencoderKL(
    in_channels=1,
    out_channels=1,
    latent_channels=4,
    block_out_channels=(128, 256, 512),
    norm_num_groups=32,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

dataset = PairedResDataset("your_paths.csv")
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

for epoch in range(10):
    for lowres, highres in dataloader:
        lowres = lowres.to(device)
        highres = highres.to(device)

        # Forward
        latents = model.encode(lowres).latent_dist.sample()
        recon = model.decode(latents).sample

        # Loss & backward
        loss = loss_fn(recon, highres)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
