import os
import nrrd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from diffusers import DDIMScheduler, UNet2DModel
from tqdm import tqdm
from torch.optim import Adam
from datetime import datetime
import torch.nn as nn
from diffusers import DDIMScheduler
import pandas as pd


device = torch.device("cuda")
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("PyTorch version:", torch.__version__)

parquet = "/mnt/raid/C1_ML_Analysis/CSV_files/extract_frames_Dataset_C_masked_resampled_256_spc075_wscores_meta_noflyto_1e-4.parquet"
df = pd.read_parquet(parquet, engine="pyarrow")
root_dir = '/mnt/raid/C1_ML_Analysis'

class DatasetFromDataFrame(Dataset):
    def __init__(self, root_dir, dataframe, transform=None):
        self.root_dir = root_dir
        self.df = dataframe
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for i in range(df.shape[0]):
            img_name = df.iloc[i,0]
            self.image_paths.append(os.path.join(self.root_dir,img_name))
            self.labels.append(0)
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        data, header = nrrd.read(img_path)
        data = np.squeeze(data)
        normalized = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-5)
        scaled_data = (normalized * 255)
        scaled_data = scaled_data[0].astype(np.uint8)
        image = Image.fromarray(scaled_data).convert('RGB')
        image = image.rotate(270)
    

        if self.transform:
            image = self.transform(image)
        
        label = int(self.labels[idx])

        return image, label
    
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.ToTensor(),           # Convert images to PyTorch tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize
])  

train_dataset = DatasetFromDataFrame(root_dir=root_dir,dataframe=df,transform=transform)
scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="linear")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"output_parquet/output_{timestamp}/"
os.makedirs(output_dir, exist_ok=True)

# Initialize model, scheduler, and optimizer
model = UNet2DModel()
# model = nn.DataParallel(model)
model = model.to(device)
optimizer = Adam(model.parameters(), lr=1)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="linear")


# Training loop
num_epochs = 1
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    model.train()
    for batch_idx, (x,y) in enumerate(tqdm(train_dataloader, desc="Training", leave=False)):
        x = x.to(device)
        # Sample noise and timesteps
        noise = torch.randn_like(x)
        noise = noise.to(device)
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (x.size(0),))
        timesteps=timesteps.to(device)
        # Forward pass with noisy input
        noisy_images = scheduler.add_noise(x, noise, timesteps)
        noise_pred = model(noisy_images, timesteps).sample

        # Compute loss
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    checkpoint_filename = f"checkpoint_epoch_{epoch+1}.pth"
    checkpoint_path = os.path.join(output_dir, checkpoint_filename)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }
torch.save(checkpoint, checkpoint_path)

print("Training completed.")
final_model_path = os.path.join(output_dir, "model.pth")
torch.save(model.state_dict(), final_model_path)
