from monai_model import DDPMPL
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from datetime import datetime
import pytorch_lightning as pl
from PIL import Image
from pytorch_lightning.strategies import DDPStrategy
import nrrd
import pandas as pd
import numpy as np

from pytorch_lightning.loggers import NeptuneLogger

neptune_key = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkZWUyNGYzZi05ZDE0LTQwYjAtYTQzOS04M2QxZmQ5MTQ0MjcifQ=="


neptune_logger = NeptuneLogger(
    api_key=neptune_key,  # Or use os.getenv("NEPTUNE_API_TOKEN")
    project="alexandrejarry/data-synthesis",  # Replace with your actual project name
)

device = torch.device("cuda")
parquet = "/mnt/raid/C1_ML_Analysis/CSV_files/extract_frames_Dataset_C_masked_resampled_256_spc075_wscores_meta_noflyto_1e-4.parquet"
df = pd.read_parquet(parquet, engine="pyarrow")

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

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = [item for item in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir,item))]
        self.image_paths = []
        self.labels = []
        
        # Collect image file paths and corresponding labels
        for label in self.classes:
            class_dir = os.path.join(root_dir, label)
            for img_name in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, img_name))
                self.labels.append(self.classes.index(label))  # Assign numeric labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # Load the image
        if self.transform:
            image = self.transform(image)

        label = int(self.labels[idx])       # Get the label
        return image, label

# Data preparation
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),

])
# Specify the path to your dataset
val_root_dir = '/mnt/raid/home/ajarry/data/trainer'
train_root_dir = '/mnt/raid/C1_ML_Analysis'
val_dir =  os.path.join(val_root_dir,"temp_train")

train_dataset = DatasetFromDataFrame(root_dir=train_root_dir,dataframe=df,transform=transform)
val_dataset = CustomImageDataset(root_dir=val_dir, transform=transform)


# Dataloader (you can mess with batch size)
batch_size = 8
num_workers = 4
train_dataloader = DataLoader(train_dataset, batch_size=batch_size,num_workers=num_workers, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size,num_workers=num_workers, shuffle=False)

# Define hyperparameters
hparams = {
    'lr': 1e-4,
    'weight_decay': 1e-5,
    'num_train_timesteps': 1000,
    'in_channels': 3,
    'out_channels': 3,
    'use_pre_trained': False
}
guidance_image_path = "/mnt/raid/home/ajarry/data/ultrasound.png"
model = DDPMPL(guidance_path=guidance_image_path,**hparams)
model.to(device)

# Set up the trainer
trainer = pl.Trainer(logger=neptune_logger, max_epochs=1, precision=16, enable_progress_bar=True, strategy=DDPStrategy(find_unused_parameters=True), log_every_n_steps=100)

# Train the model
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"outputs_monai/output_{timestamp}/"
os.makedirs(output_dir, exist_ok=True)

final_model_path = os.path.join(output_dir, "model.pth")
torch.save(model.state_dict(), final_model_path)