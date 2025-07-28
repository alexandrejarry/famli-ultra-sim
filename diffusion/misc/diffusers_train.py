import torch
import torchvision
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from matplotlib import pyplot as plt
import os
from datetime import datetime
from PIL import Image
from diffusers import UNet2DModel, DDPMScheduler
import pandas as pd
import nrrd
import numpy as np
import tqdm
torchvision.disable_beta_transforms_warning()



#####################################################################################################################################################################

class DatasetFromDataFrame(Dataset):
    def __init__(self, root_dir, dataframe, transform=None):
        self.root_dir = root_dir
        self.df = dataframe
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for i in range(self.df.shape[0]):
            img_name = self.df.iloc[i,0]
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
    
#####################################################################################################################################################################

# Define the training function
def train(rank, world_size, model, train_loader, val_loader, optimizer, scheduler, num_epochs):

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    print("MASTER_ADDR")
    
    os.environ['MASTER_PORT'] = '12355'
    print("MASTER_PORT")

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    print("process initialized")

    print("train start")
    # Move the model to the current GPU
    torch.cuda.set_device(rank)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    # Training loop
    for epoch in range(tqdm(num_epochs)):
        model.train()
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(rank)
            optimizer.zero_grad()

            # Forward pass
            noise = torch.randn_like(data)
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (data.shape[0],), device=rank).long()
            noisy_data = scheduler.add_noise(data, noise, timesteps)
            output = model(noisy_data, timesteps).sample

            # Compute loss
            loss = torch.nn.functional.mse_loss(output, noise)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0 and rank == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item()}")
            
        # Validation
        if rank == 0:  # Only validate on the main process
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data, _ in val_loader:
                    data = data.to(rank)
                    noise = torch.randn_like(data)
                    timesteps = torch.randint(0, scheduler.num_train_timesteps, (data.shape[0],), device=rank).long()
                    noisy_data = scheduler.add_noise(data, noise, timesteps)
                    output = model(noisy_data, timesteps).sample
                    val_loss += torch.nn.functional.mse_loss(output, noise).item()

            val_loss /= len(val_loader)
            print(f"Epoch [{epoch}/{num_epochs}], Validation Loss: {val_loss}")

    dist.destroy_process_group()

#####################################################################################################################################################################

# Define the main function
def main(rank,world_size):
    device = torch.device("cuda")
    print(f"Using device: {device}")
    print("main called")
    torchvision.disable_beta_transforms_warning()

    # Hyperparameters
    batch_size = 8
    num_epochs = 1
    learning_rate = 1e-4
    world_size = 4  # Number of GPUs
    num_workers = 4

    # Initialize the model and scheduler
    model = UNet2DModel(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    )
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Load the dataset
    transform = transforms.Compose([
    transforms.Resize((256,256)),  
    transforms.ToTensor(),  
    ])  

    train_root_dir = '/mnt/raid/C1_ML_Analysis'
    parquet = "/mnt/raid/C1_ML_Analysis/CSV_files/extract_frames_Dataset_C_masked_resampled_256_spc075_wscores_meta_noflyto_1e-4.parquet"
    df = pd.read_parquet(parquet, engine="pyarrow")

    full_dataset = DatasetFromDataFrame(root_dir=train_root_dir,dataframe=df,transform=transform)

    train_ratio = 0.8
    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    # Launch the training processes
    train(rank, world_size, model, train_loader,val_loader, optimizer, scheduler, num_epochs)

if __name__ == "__main__":
    world_size = 4
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)