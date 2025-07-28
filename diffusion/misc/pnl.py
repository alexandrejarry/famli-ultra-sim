import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from torchvision.io import read_image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import os
from datetime import datetime
from PIL import Image
from diffusers import UNet2DModel, DDIMScheduler
from tqdm import tqdm
import neptune

import torchvision.utils as vutils

device = torch.device("cuda:1")
neptune_key = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkZWUyNGYzZi05ZDE0LTQwYjAtYTQzOS04M2QxZmQ5MTQ0MjcifQ=="
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
        # image = image.float() / 255.0

        if self.transform:
            image = self.transform(image)

        label = int(self.labels[idx])       # Get the label
        

        
        return image, label

transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.ToTensor(),           # Convert images to PyTorch tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize
])

# Specify the path to your dataset
root_dir = '/mnt/raid/home/ajarry/data/trainer'
train_dir = os.path.join(root_dir,"temp_train/")
val_dir = os.path.join(root_dir,"temp_val/")

# data_dir = '/mnt/famli_netapp_shared/C1_ML_Analysis/famli_ml_lists/AnalysisLists/East/datasets'
# train_dir = os.path.join(data_dir, 'frame_C2_bfly_Head_ordinal_boundary/train/')
# val_dir =  os.path.join(root_dir,"temp_train")

# Create dataset instance
# custom_dataset = CustomImageDataset(root_dir=root_dir, transform=transform)
train_dataset = CustomImageDataset(root_dir=train_dir,transform=transform)
val_dataset = CustomImageDataset(root_dir=val_dir,transform=transform)

class BasicUNet(nn.Module):
    """A minimal UNet implementation."""

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.down_layers = torch.nn.ModuleList(
            [
                nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
                nn.Conv2d(32, 64, kernel_size=5, padding=2),
                nn.Conv2d(64, 64, kernel_size=5, padding=2),
            ]
        )
        self.up_layers = torch.nn.ModuleList(
            [
                nn.Conv2d(64, 64, kernel_size=5, padding=2),
                nn.Conv2d(64, 32, kernel_size=5, padding=2),
                nn.Conv2d(32, out_channels, kernel_size=5, padding=2),
            ]
        )
        self.act = nn.ReLU()  # The activation function
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)

    def forward(self, x):
        h = []
        for i, l in enumerate(self.down_layers):
            x = self.act(l(x))  # Through the layer and the activation function
            if i < 2:  # For all but the third (final) down layer:
                h.append(x)  # Storing output for skip connection
                x = self.downscale(x)  # Downscale ready for the next layer

        for i, l in enumerate(self.up_layers):
            if i > 0:  # For all except the first up layer
                x = self.upscale(x)  # Upscale
                x += h.pop()  # Fetching stored output (skip connection)
            x = self.act(l(x))  # Through the layer and the activation function

        return x

def compute_loss(self, x, step='train', sync_dist=False):
    """
    Args:
        x:  Input point clouds, (B, N, d).
    """
    batch_size, _, _ = x.size()
    
    h = self.encoder(x.permute(0, 2, 1))
    z_mu = self.proj_mu(h)
    z_sigma = self.proj_sigma(h)
    
    # z_mu, z_sigma = self.encoder(x, x)
    # z_mu = z_mu.mean(dim=1)
    # z_sigma = z_sigma.mean(dim=1)
    
    z = self.reparameterize_gaussian(mean=z_mu, logvar=z_sigma)  # (B, F)
    
    # H[Q(z|X)]
    entropy = self.gaussian_entropy(logvar=z_sigma)      # (B, )
    loss_entropy = -entropy.mean()
    # loss_entropy = 0.0

    # P(z), Prior probability, parameterized by the flow: z -> w.
    
    # w, delta_log_pw = self.flow(z, torch.zeros([batch_size, 1]).to(z), reverse=False)
    # log_pw = self.standard_normal_logprob(w).view(batch_size, -1).sum(dim=1, keepdim=True)   # (B, 1)
    # log_pz = log_pw - delta_log_pw.view(batch_size, 1)  # (B, 1)
    # loss_prior = -log_pz.mean()

    loss_prior = -self.flow.log_prob(inputs=z).mean()

    # Negative ELBO of P(X|z)
    noise = torch.randn_like(x)
    x_noisy, beta = self.noise_scheduler.add_noise(x, noise=noise)
    x_pred = self.diffusion(x_noisy, beta=beta, context=z)
    loss_recons = F.mse_loss(x_pred, noise, reduction='mean')

    # neg_elbo = self.diffusion.get_loss(x, z)
    # loss_recons = neg_elbo

    # Loss
    
    loss = self.hparams.kl_weight*(loss_entropy + loss_prior) + loss_recons

    self.log(f"{step}_loss", loss, sync_dist=sync_dist)
    self.log(f"{step}_loss_prior", loss_prior, sync_dist=sync_dist)
    self.log(f"{step}_loss_recons", loss_recons, sync_dist=sync_dist)
    self.log(f"{step}_z_mean", z_mu.mean(), sync_dist=sync_dist)
    self.log(f"{step}_z_mag", z_mu.abs().max(), sync_dist=sync_dist)
    self.log(f"{step}_z_var", (0.5*z_sigma).exp().mean(), sync_dist=sync_dist)
    # self.log(f"{step}_x_pred_mean", x_pred.mean(), sync_dist=sync_dist)
    # self.log(f"{step}_x_pred_std", x_pred.std(), sync_dist=sync_dist)

    return loss

# Dataloader (you can mess with batch size)
batch_size = 32
num_workers = 4
train_dataloader = DataLoader(train_dataset, batch_size=batch_size,num_workers=num_workers, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size,num_workers=num_workers, shuffle=True)
scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="linear")

# How many runs through the data should we do?
n_epochs = 10

# Create the network
net = UNet2DModel()
net.to(device)


# Our loss function
loss_fn = nn.MSELoss()

# The optimizer
opt = torch.optim.Adam(net.parameters(), lr=1e-1)

# Keeping a record of the losses for later viewing
train_losses = []
val_losses = []

# Define directory for saving checkpoints
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"outputs_v2/output_{timestamp}/"
os.makedirs(output_dir, exist_ok=True)

run = neptune.init_run(
    project="alexandrejarry/data-synthesis",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkZWUyNGYzZi05ZDE0LTQwYjAtYTQzOS04M2QxZmQ5MTQ0MjcifQ=="
)
params = {"learning_rate": 1, "optimizer": "Adam"}
run["parameters"] = params

# The training loop
for epoch in range(n_epochs):

    net.train()
    
    for batch_idx, (x,y) in enumerate(tqdm(train_dataloader, desc="Training", leave=False)):
        # Get some data and prepare the corrupted version
        x = x.to(device)  # Data on the GPU
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (x.size(0),)).to(device)

        noise = torch.rand_like(x)
        noise = noise.to(device)

        # Get the model prediction
        noisy_images = scheduler.add_noise(x, noise, timesteps)
        pred = net(noisy_images, timesteps).sample

        # Calculate the loss
        train_loss = loss_fn(pred,x)  # How close is the output to the true 'clean' x?

        # Backprop and update the params:
        opt.zero_grad()
        train_loss.backward()
        opt.step()

        # Store the loss for later
        train_losses.append(train_loss.item())
        run["epoch"].append(epoch)
        run["train?loss"].append(train_loss.item())

    # Validation phase
    net.eval()
    with torch.no_grad():
        for x, y in val_dataloader:
            x = x.to(device)
            y = y.to(device)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (x.size(0),), device=device)
            outputs = net(x,timesteps).sample
            val_loss = loss_fn(outputs,x)
            val_losses.append(val_loss.item())
            run["validation loss"].append(val_loss.item())
            

    # Print our the average of the loss values for this epoch:
    avg_train_loss = sum(train_losses[-len(train_dataloader) :]) / len(train_dataloader)
    avg_val_loss = sum(val_losses) / len(val_dataloader)
    print(f"Finished epoch {epoch}. Average loss for this epoch: {avg_train_loss:05f}. Average validation loss: {avg_val_loss:05f}")
    checkpoint_filename = f"checkpoint_epoch_{epoch+1}.pth"
    checkpoint_path = os.path.join(output_dir, checkpoint_filename)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        }
    torch.save(checkpoint, checkpoint_path)
run.stop()
# View the loss curve
plt.plot(train_losses)
final_model_path = os.path.join(output_dir, "model.pth")
torch.save(net.state_dict(), final_model_path)
