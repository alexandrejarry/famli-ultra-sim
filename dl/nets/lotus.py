import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

import lightning as L
from lightning.pytorch.core import LightningModule
import numpy as np


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

class UltrasoundRendering(LightningModule):
    def __init__(self, **kwargs): 
        super().__init__()

        self.save_hyperparameters()
        # df = pd.read_csv(acoustic_params_fn)        
        # accoustic_imped,attenuation,mu_0,mu_1,sigma_0
        self.acoustic_impedance_dict = torch.nn.Parameter(torch.randn(self.hparams.num_labels)*5.0 + 5.0)    # Z in MRayl
        self.attenuation_dict =    torch.nn.Parameter(torch.randn(self.hparams.num_labels) + 1.0)   # alpha in dB cm^-1 at 1 MHz
        self.mu_0_dict =           torch.nn.Parameter(torch.randn(self.hparams.num_labels)*0.5 + 0.5) # mu_0 - scattering_mu   mean brightness
        self.mu_1_dict =           torch.nn.Parameter(torch.randn(self.hparams.num_labels)*0.5 + 0.5) # mu_1 - scattering density, Nr of scatterers/voxel
        self.sigma_0_dict =        torch.nn.Parameter(torch.randn(self.hparams.num_labels)*0.5 + 0.5) # sigma_0 - scattering_sigma - brightness std
        
        grid, inverse_grid, mask = self.init_grids(self.hparams.grid_w, self.hparams.grid_h, self.hparams.center_x, self.hparams.center_y, self.hparams.r1, self.hparams.r2, self.hparams.theta)

        self.register_buffer("grid", grid)
        self.register_buffer("inverse_grid", inverse_grid)
        self.register_buffer("mask_fan", mask)

        g_kernel = self.gaussian_kernel_asym(3, 0., 0.5)[None, None, :, :]
        self.register_buffer("g_kernel", g_kernel)

        self.loss = nn.L1Loss()

    @staticmethod
    def add_model_specific_args(parent_parser):
        hparams_group = parent_parser.add_argument_group(title="Ultrasound Rendering")
        hparams_group.add_argument('--num_labels', help='Number of labels in the US model', type=int, default=12)
        hparams_group.add_argument('--grid_w', help='Grid size for the simulation', type=int, default=256)
        hparams_group.add_argument('--grid_h', help='Grid size for the simulation', type=int, default=256)
        hparams_group.add_argument('--center_x', help='Position of the circle that creates the transducer', type=float, default=128.0)
        hparams_group.add_argument('--center_y', help='Position of the circle that creates the transducer', type=float, default=-30.0)
        hparams_group.add_argument('--r1', help='Radius of first circle', type=float, default=20.0)
        hparams_group.add_argument('--r2', help='Radius of second circle', type=float, default=215.0)
        hparams_group.add_argument('--theta', help='Aperture angle of transducer', type=float, default=np.pi/4.0)
        hparams_group.add_argument('--alpha_coeff_boundary_map', help='Lotus model', type=float, default=0.1)
        hparams_group.add_argument('--beta_coeff_scattering', help='Lotus model', type=float, default=10)
        hparams_group.add_argument('--tgc', help='Lotus model', type=int, default=8)
        hparams_group.add_argument('--clamp_vals', help='Lotus model', type=int, default=1)

        return parent_parser

        
    def init_params(self, df):
        # df = pd.read_csv(acoustic_params_fn)
        
        # accoustic_imped,attenuation,mu_0,mu_1,sigma_0
        self.acoustic_impedance_dict = torch.nn.Parameter(torch.tensor(df['acoustic_impedance_dict'], dtype=torch.float32))    # Z in MRayl
        self.attenuation_dict =    torch.nn.Parameter(torch.tensor(df['attenuation_dict'], dtype=torch.float32))   # alpha in dB cm^-1 at 1 MHz
        self.mu_0_dict =           torch.nn.Parameter(torch.tensor(df['mu_0_dict'], dtype=torch.float32)) # mu_0 - scattering_mu   mean brightness
        self.mu_1_dict =           torch.nn.Parameter(torch.tensor(df['mu_1_dict'], dtype=torch.float32)) # mu_1 - scattering density, Nr of scatterers/voxel
        self.sigma_0_dict =        torch.nn.Parameter(torch.tensor(df['sigma_0_dict'], dtype=torch.float32)) # sigma_0 - scattering_sigma - brightness std

    def init_grids(self, w, h, center_x, center_y, r1, r2, theta):
        grid = self.compute_grid(w, h, center_x, center_y, r1, r2, theta)
        inverse_grid, mask = self.compute_grid_inverse(grid)
        grid = self.normalize_grid(grid)
        inverse_grid = self.normalize_grid(inverse_grid)

        return  grid, inverse_grid, mask

    def compute_grid(self, w, h, center_x, center_y, r1, r2, theta):

        # Convert inputs to tensors
        angles = torch.linspace(-theta, theta, w)  # Angles from -theta to theta
        radii = torch.linspace(r1, r2, h)  # Linear space of radii

        # Calculate sin and cos for all angles (broadcasting)
        sin_angles = torch.sin(angles)
        cos_angles = torch.cos(angles)

        # Initialize the grid for intersection points
        # Shape of grid: (h, w, 2) where 2 represents (x, y) coordinates
        grid = torch.zeros(h, w, 2)

        # Calculate intersections for each radius and angle
        for i, radius in enumerate(radii):
            x = (center_x + radius * sin_angles) # x coordinates for all angles at this radius
            y = (center_y + radius * cos_angles) # y coordinates for all angles at this radius

            grid[i] = torch.stack((x, y), dim=1)  # Update grid with coordinates

        return grid
        

    def compute_grid_inverse(self, grid):

        h, w, _ = grid.shape  # grid dimensions
        inverse_grid = torch.zeros(h, w, 2)  # Initialize inverse grid
        mask = torch.zeros(1, h, w)  # Initialize mask

        # Iterate through each point in the grid
        for j in range(h):
            for i in range(w):
                # Extract the polar coordinates (represented in the grid)
                xi, yi = torch.round(grid[j, i]).to(torch.long)

                # Place the Cartesian coordinates in the inverse grid
                if 0 <= xi and xi < w and 0 <= yi and yi < h:
                    inverse_grid[yi, xi] = torch.tensor([i, j])
                    mask[0, yi, xi] = 1
        return inverse_grid, self.morphology_close(mask.unsqueeze(0)).squeeze(0)

    def normalize_grid(self, grid):
        h, w, _ = grid.shape  # grid dimensions
        grid = grid / torch.tensor([h, w]) * 2.0 - 1.0
        return grid

    def dilate(self, x, kernel_size = 3):
        kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32)

        # Apply convolution to simulate dilation
        # We use padding=1 to ensure the output size is the same as the input size
        output = F.conv2d(x, kernel, padding=1)

        # Apply a threshold to get a binary output
        dilated_image = (output > 0).float()

        return dilated_image
    
    def erode(self, x, kernel_size = 3):
        # Step 2: Erosion
        # For erosion, invert the image and kernel, apply dilation, then invert the output
        x = 1 - x
        inverted_kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32) # Same kernel as for dilation

        # Apply convolution (dilation on inverted image) with padding to maintain size
        eroded_output_inverted = F.conv2d(x, inverted_kernel, padding=1)

        # Invert the result to get the final eroded (closing) result
        eroded_image = 1 - (eroded_output_inverted > 0).float()

        return eroded_image

    def morphology_close(self, x, kernel_size=3):
        return self.erode(self.dilate(x, kernel_size), kernel_size)

    def add_speckle_noise(self, x, noise_variance=0.1):
        """
        Adds speckle noise to an image.

        Parameters:
        - x: A PyTorch tensor representing the image, shape (C, H, W) or (B, C, H, W)
        - noise_variance: Variance of the Gaussian noise

        Returns:
        - Noisy image: A PyTorch tensor of the same shape as `image` with speckle noise added
        """
        # Ensure the noise is generated for each image in the batch
        if x.dim() == 3: # For single image (C, H, W)
            noise_shape = x.shape
        elif x.dim() == 4: # For batch of images (B, C, H, W)
            noise_shape = (x.size(0), 1, x.size(2), x.size(3))

        # Generate noise from a normal distribution centered at 1 with specified variance
        noise = 1 + torch.randn(noise_shape, device=x.device) * noise_variance

        # Add noise to the image
        return x * noise

    
    def gaussian_kernel_sym(self, size, sigma):
        """
        Creates a 2D Gaussian kernel.

        Args:
        size (int): The size of the kernel.
        sigma (float): The standard deviation of the Gaussian distribution.

        Returns:
        Tensor: A 2D Gaussian kernel.
        """
        coords = torch.arange(size, dtype=torch.float32)
        coords -= size // 2

        g = coords**2
        g = (-g / (2 * sigma**2)).exp()

        g /= g.sum()
        
        gaussian = g.outer(g)
        
        return gaussian
    
    def gaussian_kernel_asym(self, size: int, mean: float, std: float):
        d1 = torch.distributions.Normal(mean, std)
        d2 = torch.distributions.Normal(mean, std*3)
        vals_x = d1.log_prob(torch.arange(-size, size+1, dtype=torch.float32)).exp()
        vals_y = d2.log_prob(torch.arange(-size, size+1, dtype=torch.float32)).exp()

        gauss_kernel = torch.einsum('i,j->ij', vals_x, vals_y)

        return gauss_kernel / torch.sum(gauss_kernel).reshape(1, 1)


    def smooth(self, x, kernel_size=3, sigma=1, maxpool=True):
        """
        Applies Gaussian blur to a batch of images.

        Args:
        images (Tensor): Input images with shape (N, C, H, W).
        kernel_size (int): The size of the Gaussian kernel.
        sigma (float): The standard deviation of the Gaussian distribution.

        Returns:
        Tensor: The smoothed images.
        """
        # Create a Gaussian kernel
        kernel = self.gaussian_kernel_sym(kernel_size, sigma).to(x.device)
        
        kernel = kernel.expand(x.size(1), 1, kernel_size, kernel_size)

        # Apply the Gaussian kernel to each image in the batch
        padding = kernel_size // 2
        
        if maxpool:
            x = torch.nn.MaxPool2d(kernel_size, stride=1, padding=1)(x)
            
        return F.conv2d(x, kernel, padding=padding, groups=x.size(1))
    
    def rendering(self, shape, attenuation_medium_map, mu_0_map, mu_1_map, sigma_0_map, z_vals=None, refl_map=None, boundary_map=None):
        
        dists = torch.abs(z_vals[..., :-1, None] - z_vals[..., 1:, None])     # dists.shape=(W, H-1, 1)
        dists = dists.squeeze(-1)                                             # dists.shape=(W, H-1)
        dists = torch.cat([dists, dists[:, -1, None]], dim=-1)                # dists.shape=(W, H)
        
        attenuation = torch.exp(-attenuation_medium_map * dists)
        attenuation_total = torch.cumprod(attenuation, dim=3, dtype=torch.float32, out=None)

        gain_coeffs = torch.linspace(1, self.hparams.tgc, attenuation_total.shape[3], device=self.device)
        gain_coeffs = torch.tile(gain_coeffs, (attenuation_total.shape[2], 1))
        attenuation_total = attenuation_total * gain_coeffs     # apply TGC

        reflection_total = torch.cumprod(1. - refl_map * boundary_map, dim=3, dtype=torch.float32, out=None) 
        reflection_total = reflection_total.squeeze(-1) 
        reflection_total_plot = torch.log(reflection_total + torch.finfo(torch.float32).eps)

        texture_noise = torch.randn(shape, dtype=torch.float32, device=self.device)
        scattering_probability = torch.randn(shape, dtype=torch.float32, device=self.device)

        # scattering_zero = torch.zeros(shape, dtype=torch.float32)

        z = mu_1_map - scattering_probability
        sigmoid_map = torch.sigmoid(self.hparams.beta_coeff_scattering * z)

        # approximating  Eq. (4) to be differentiable:
        # where(scattering_probability <= mu_1_map, 
        #                     texture_noise * sigma_0_map + mu_0_map, 
        #                     scattering_zero)
        # scatterers_map =  (sigmoid_map) * (texture_noise * sigma_0_map + mu_0_map) + (1 -sigmoid_map) * scattering_zero   # Eq. (6)
        scatterers_map =  (sigmoid_map) * (texture_noise * sigma_0_map + mu_0_map)

        psf_scatter_conv = torch.nn.functional.conv2d(input=scatterers_map, weight=self.g_kernel, stride=1, padding="same")
        # psf_scatter_conv = psf_scatter_conv.squeeze()

        b = attenuation_total * psf_scatter_conv    # Eq. (3)

        border_convolution = torch.nn.functional.conv2d(input=boundary_map, weight=self.g_kernel, stride=1, padding="same")
        # border_convolution = border_convolution.squeeze()

        r = attenuation_total * reflection_total * refl_map * border_convolution # Eq. (2)
        
        intensity_map = b + r   # Eq. (1)
        # intensity_map = intensity_map.squeeze() 
        intensity_map = torch.clamp(intensity_map, 0, 1)

        return intensity_map, attenuation_total, reflection_total_plot, scatterers_map, scattering_probability, border_convolution, texture_noise, b, r
    
    def render_rays(self, W, H, device='cuda'):
        N_rays = W 
        t_vals = torch.linspace(0., 1., H, device=device)
        z_vals = t_vals.unsqueeze(0).expand(N_rays , -1) * 4 

        return z_vals

    def forward(self, x, grid=None, inverse_grid=None, mask_fan=None, return_seg=False):

        if grid is None:

            #init tissue maps
            #generate maps from the dictionary and the input label map
            repeats = [1,]*len(x.shape)
            repeats[0] = x.shape[0]

            grid = self.grid
            inverse_grid = self.inverse_grid
            mask_fan = self.mask_fan

            grid = grid.repeat(repeats)
            inverse_grid = inverse_grid.repeat(repeats)
            mask_fan = mask_fan.repeat(repeats)
        
        if grid.shape[0] != x.shape[0]:
            repeats = [1,]*len(x.shape)
            repeats[0] = x.shape[0]
            grid = grid.repeat(repeats)
            inverse_grid = inverse_grid.repeat(repeats)
            mask_fan = mask_fan.repeat(repeats)

        #UNWARP
        x = F.grid_sample(x.float(), grid, mode='nearest', padding_mode='zeros', align_corners=True)

        x = torch.rot90(x, k=1, dims=[2, 3])
        x = x.to(torch.long)
        
        acoustic_imped_map = self.acoustic_impedance_dict[x]
        attenuation_medium_map = self.attenuation_dict[x]
        mu_0_map = self.mu_0_dict[x]
        mu_1_map = self.mu_1_dict[x]
        sigma_0_map = self.sigma_0_dict[x]

        if hasattr(self.hparams, 'clamp_vals') and self.hparams.clamp_vals:
            acoustic_imped_map = torch.clamp(acoustic_imped_map, 0, 10)
            attenuation_medium_map = torch.clamp(attenuation_medium_map, 0, 10)
            sigma_0_map = torch.clamp(sigma_0_map, 0, 1)
            mu_1_map = torch.clamp(mu_1_map, 0, 1)
            mu_0_map = torch.clamp(mu_0_map, 0, 1)

        
        #Comput the difference along dimension 2
        diff_arr = torch.diff(acoustic_imped_map, dim=2)                
        # The pad tuple is (padding_left,padding_right, padding_top,padding_bottom)
        # The array is padded at the top
        diff_arr = F.pad(diff_arr, (0,0,1,0))

        #Compute the boundary map using the diff_array
        boundary_map =  -torch.exp(-(diff_arr**2)/self.hparams.alpha_coeff_boundary_map) + 1
        
        #Roll/shift the elements along dimension 2 and set the last element to 0
        shifted_arr = torch.roll(acoustic_imped_map, -1, dims=2)
        shifted_arr[-1:] = 0

        # This computes the sum/accumulation along the direction and set elements that are 0 to 1. Compute the division
        sum_arr = acoustic_imped_map + shifted_arr
        sum_arr[sum_arr == 0] = 1
        div = diff_arr / sum_arr
        # Compute the reflection from the elements
        refl_map = div ** 2
        refl_map = torch.sigmoid(refl_map)      # 1 / (1 + (-refl_map).exp())

        z_vals = self.render_rays(x.shape[2], x.shape[3], device=x.device)

        ret_list = self.rendering(x.shape, attenuation_medium_map, mu_0_map, mu_1_map, sigma_0_map, z_vals=z_vals, refl_map=refl_map, boundary_map=boundary_map)

        intensity_map  = ret_list[0]

        x = torch.rot90(x, k=3, dims=[2, 3])
        intensity_map = torch.rot90(intensity_map, k=3, dims=[2, 3])
        
        x = F.grid_sample(x.float(), inverse_grid, mode='nearest', padding_mode='zeros', align_corners=True).long()
        intensity_map = F.grid_sample(intensity_map.float(), inverse_grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        # return intensity_map, x, attenuation_medium_map, mu_0_map, mu_1_map, sigma_0_map, acoustic_imped_map, boundary_map, shifted_arr
        
        intensity_map = intensity_map * mask_fan

        #intensity_map_s = self.smooth(intensity_map)
        #intensity_map[mask_fan==0] = intensity_map_s[mask_fan==0]

        # return intensity_map, x, attenuation_medium_map, mu_0_map, mu_1_map, sigma_0_map, acoustic_imped_map, boundary_map, shifted_arr
        if return_seg:
            return intensity_map, x
        return intensity_map
    
    
    
    def configure_optimizers(self):        
        
        # optimizer = optim.AdamW(self.parameters(),
        #                         lr=self.hparams.lr,
        #                         weight_decay=self.hparams.weight_decay)
        optimizer = optim.SGD(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay, momentum=self.hparams.momentum)
        
        return optimizer
    

    def training_step(self, train_batch, batch_idx):
        seg = train_batch['seg']
        img = train_batch['img']

        fake_us = self(seg)[0]

        repeats = [1,]*len(img.shape)
        repeats[0] = img.shape[0]
        mask_fan = self.mask_fan.repeat(repeats)
        img = mask_fan*img

        loss = self.loss(fake_us, img)

        self.log("loss", loss)

        return loss
        
    
    def validation_step(self, val_batch, batch_idx):        
        seg = val_batch['seg']
        img = val_batch['img']

        fake_us = self(seg)[0]

        repeats = [1,]*len(img.shape)
        repeats[0] = img.shape[0]
        mask_fan = self.mask_fan.repeat(repeats)
        img = mask_fan*img

        val_loss = self.loss(fake_us, img)

        self.log("val_loss", val_loss, sync_dist=True)

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initializes the projection head.

        Parameters:
        - input_dim: Dimensionality of the input features.
        - hidden_dim: Dimensionality of the hidden layer.
        - output_dim: Dimensionality of the output features (projected space).
        """
        super(ProjectionHead, self).__init__()
        
        # Define the projection head architecture
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        Forward pass of the projection head.

        Parameters:
        - x: Input features tensor.

        Returns:
        - The projected features tensor.
        """
        return self.layers(x)

class UltrasoundRenderingLinear(LightningModule):
    def __init__(self, **kwargs): 
        super().__init__()

        self.save_hyperparameters()
        
        mean_diffusor_dict = torch.rand(self.hparams.num_labels)
        variance_diffusor_dict = torch.rand(self.hparams.num_labels)

        self.register_buffer("mean_diffusor_dict", mean_diffusor_dict)
        self.register_buffer("variance_diffusor_dict", variance_diffusor_dict)

        # self.mlp_w = MLPBlock(hidden_size=self.hparams.grid_w, mlp_dim=self.hparams.mlp_dim)
        # self.mlp_h = MLPBlock(hidden_size=self.hparams.grid_h, mlp_dim=self.hparams.mlp_dim)

        # self.unet = UNet(spatial_dims=2,
        #     in_channels=2,
        #     out_channels=1,
        #     channels=(4, 8, 16),
        #     strides=(2, 2),
        #     num_res_units=2)
        
        grid, inverse_grid, mask = self.init_grids(self.hparams.grid_w, self.hparams.grid_h, self.hparams.center_x, self.hparams.center_y, self.hparams.r1, self.hparams.r2, self.hparams.theta)

        self.register_buffer("grid", grid)
        self.register_buffer("inverse_grid", inverse_grid)
        self.register_buffer("mask_fan", mask)

    @staticmethod
    def add_model_specific_args(parent_parser):
        hparams_group = parent_parser.add_argument_group(title="Ultrasound Rendering Linear")
        hparams_group.add_argument('--num_labels', help='Number of labels in the US model', type=int, default=340)
        hparams_group.add_argument('--grid_w', help='Grid size for the simulation', type=int, default=256)
        hparams_group.add_argument('--grid_h', help='Grid size for the simulation', type=int, default=256)
        hparams_group.add_argument('--center_x', help='Position of the circle that creates the transducer', type=float, default=128.0)
        hparams_group.add_argument('--center_y', help='Position of the circle that creates the transducer', type=float, default=-30.0)
        hparams_group.add_argument('--r1', help='Radius of first circle', type=float, default=10.0)
        hparams_group.add_argument('--r2', help='Radius of second circle', type=float, default=225.0)
        hparams_group.add_argument('--theta', help='Aperture angle of transducer', type=float, default=np.pi/4.75)

        return parent_parser

        
    def init_params(self, mean_diffusor_dict, variance_diffusor_dict):
        # self.mean_diffusor_dict = torch.nn.Parameter(mean_diffusor_dict)
        self.mean_diffusor_dict = torch.tensor(mean_diffusor_dict)
        self.variance_diffusor_dict =  torch.nn.Parameter(variance_diffusor_dict)

    def init_grids(self, w, h, center_x, center_y, r1, r2, theta):
        grid = self.compute_grid(w, h, center_x, center_y, r1, r2, theta)
        inverse_grid, mask = self.compute_grid_inverse(grid)
        grid = self.normalize_grid(grid)
        inverse_grid = self.normalize_grid(inverse_grid)

        return  grid, inverse_grid, mask

    def compute_grid(self, w, h, center_x, center_y, r1, r2, theta):

        # Convert inputs to tensors
        angles = torch.linspace(-theta, theta, w)  # Angles from -theta to theta
        radii = torch.linspace(r1, r2, h)  # Linear space of radii

        # Calculate sin and cos for all angles (broadcasting)
        sin_angles = torch.sin(angles)
        cos_angles = torch.cos(angles)

        # Initialize the grid for intersection points
        # Shape of grid: (h, w, 2) where 2 represents (x, y) coordinates
        grid = torch.zeros(h, w, 2)

        # Calculate intersections for each radius and angle
        for i, radius in enumerate(radii):
            x = (center_x + radius * sin_angles) # x coordinates for all angles at this radius
            y = (center_y + radius * cos_angles) # y coordinates for all angles at this radius

            grid[i] = torch.stack((x, y), dim=1)  # Update grid with coordinates

        return grid
        
        

    def compute_grid_inverse(self, grid):

        h, w, _ = grid.shape  # grid dimensions
        inverse_grid = torch.zeros(h, w, 2)  # Initialize inverse grid
        mask = torch.zeros(1, h, w)  # Initialize mask

        # Iterate through each point in the grid
        for j in range(h):
            for i in range(w):
                # Extract the polar coordinates (represented in the grid)
                xi, yi = torch.round(grid[j, i]).to(torch.long)

                # Place the Cartesian coordinates in the inverse grid
                if 0 <= xi and xi < w and 0 <= yi and yi < h:
                    inverse_grid[yi, xi] = torch.tensor([i, j])
                    mask[0, yi, xi] = 1
        return inverse_grid, self.morphology_close(mask.unsqueeze(0)).squeeze(0)

    def normalize_grid(self, grid):
        h, w, _ = grid.shape  # grid dimensions
        grid = grid / torch.tensor([h, w]) * 2.0 - 1.0
        return grid

    def dilate(self, x, kernel_size = 3):
        kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32)

        # Apply convolution to simulate dilation
        # We use padding=1 to ensure the output size is the same as the input size
        output = F.conv2d(x, kernel, padding=1)

        # Apply a threshold to get a binary output
        dilated_image = (output > 0).float()

        return dilated_image
    
    def erode(self, x, kernel_size = 3):
        # Step 2: Erosion
        # For erosion, invert the image and kernel, apply dilation, then invert the output
        x = 1 - x
        inverted_kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32) # Same kernel as for dilation

        # Apply convolution (dilation on inverted image) with padding to maintain size
        eroded_output_inverted = F.conv2d(x, inverted_kernel, padding=1)

        # Invert the result to get the final eroded (closing) result
        eroded_image = 1 - (eroded_output_inverted > 0).float()

        return eroded_image

    def morphology_close(self, x, kernel_size=3):
        return self.erode(self.dilate(x, kernel_size), kernel_size)

    def forward(self, x, grid=None, inverse_grid=None, mask_fan=None):

        if grid is None:

            #init tissue maps
            #generate maps from the dictionary and the input label map
            repeats = [1,]*len(x.shape)
            repeats[0] = x.shape[0]

            grid = self.grid
            inverse_grid = self.inverse_grid
            mask_fan = self.mask_fan

            grid = grid.repeat(repeats)
            inverse_grid = inverse_grid.repeat(repeats)
            mask_fan = mask_fan.repeat(repeats)
        
        x = x.to(torch.long)
        mean_diffusor = self.mean_diffusor_dict[x]
        variance_diffusor = self.variance_diffusor_dict[x]
        
        x = mean_diffusor * (1.0 + torch.randn(x.shape, device=x.device) * variance_diffusor)
        
        return x*mask_fan
    
    
    
    def configure_optimizers(self):        
        
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        
        return optimizer
    

    def training_step(self, train_batch, batch_idx):
        seg = train_batch['seg']
        img = train_batch['img']

        fake_us = self(seg)[0]

        repeats = [1,]*len(img.shape)
        repeats[0] = img.shape[0]
        mask_fan = self.mask_fan.repeat(repeats)
        img = mask_fan*img

        loss = self.loss(fake_us, img)

        self.log("loss", loss)

        return loss
        
    
    def validation_step(self, val_batch, batch_idx):        
        seg = val_batch['seg']
        img = val_batch['img']

        fake_us = self(seg)[0]

        repeats = [1,]*len(img.shape)
        repeats[0] = img.shape[0]
        mask_fan = self.mask_fan.repeat(repeats)
        img = mask_fan*img

        val_loss = self.loss(fake_us, img)

        self.log("val_loss", val_loss, sync_dist=True)

class UltrasoundRenderingLinearV2(LightningModule):
    def __init__(self, **kwargs): 
        super().__init__()

        self.save_hyperparameters()
        
        mean_diffusor_dict = torch.rand(self.hparams.num_labels)
        variance_diffusor_dict = torch.nn.Parameter(torch.rand(self.hparams.num_labels))

        self.register_buffer("mean_diffusor_dict", mean_diffusor_dict)
        self.register_buffer("variance_diffusor_dict", variance_diffusor_dict)

        # self.mlp_w = MLPBlock(hidden_size=self.hparams.grid_w, mlp_dim=self.hparams.mlp_dim)
        # self.mlp_h = MLPBlock(hidden_size=self.hparams.grid_h, mlp_dim=self.hparams.mlp_dim)

        # self.unet = UNet(spatial_dims=2,
        #     in_channels=2,
        #     out_channels=1,
        #     channels=(4, 8, 16),
        #     strides=(2, 2),
        #     num_res_units=2)
        
        grid, inverse_grid, mask = self.init_grids(self.hparams.grid_w, self.hparams.grid_h, self.hparams.center_x, self.hparams.center_y, self.hparams.r1, self.hparams.r2, self.hparams.theta)

        self.register_buffer("grid", grid)
        self.register_buffer("inverse_grid", inverse_grid)
        self.register_buffer("mask_fan", mask)

        
    def init_params(self, mean_diffusor_dict, variance_diffusor_dict):
        self.mean_diffusor_dict = mean_diffusor_dict
        # self.variance_diffusor_dict =  torch.nn.Parameter(variance_diffusor_dict)

    def init_grids(self, w, h, center_x, center_y, r1, r2, theta):
        grid = self.compute_grid(w, h, center_x, center_y, r1, r2, theta)
        inverse_grid, mask = self.compute_grid_inverse(grid)
        grid = self.normalize_grid(grid)
        inverse_grid = self.normalize_grid(inverse_grid)

        return  grid, inverse_grid, mask

    def compute_grid(self, w, h, center_x, center_y, r1, r2, theta):

        # Convert inputs to tensors
        angles = torch.linspace(-theta, theta, w)  # Angles from -theta to theta
        radii = torch.linspace(r1, r2, h)  # Linear space of radii

        # Calculate sin and cos for all angles (broadcasting)
        sin_angles = torch.sin(angles)
        cos_angles = torch.cos(angles)

        # Initialize the grid for intersection points
        # Shape of grid: (h, w, 2) where 2 represents (x, y) coordinates
        grid = torch.zeros(h, w, 2)

        # Calculate intersections for each radius and angle
        for i, radius in enumerate(radii):
            x = (center_x + radius * sin_angles) # x coordinates for all angles at this radius
            y = (center_y + radius * cos_angles) # y coordinates for all angles at this radius

            grid[i] = torch.stack((x, y), dim=1)  # Update grid with coordinates

        return grid
        

    def compute_grid_inverse(self, grid):

        h, w, _ = grid.shape  # grid dimensions
        inverse_grid = torch.zeros(h, w, 2)  # Initialize inverse grid
        mask = torch.zeros(1, h, w)  # Initialize mask

        # Iterate through each point in the grid
        for j in range(h):
            for i in range(w):
                # Extract the polar coordinates (represented in the grid)
                xi, yi = torch.round(grid[j, i]).to(torch.long)

                # Place the Cartesian coordinates in the inverse grid
                if 0 <= xi and xi < w and 0 <= yi and yi < h:
                    inverse_grid[yi, xi] = torch.tensor([i, j])
                    mask[0, yi, xi] = 1
        return inverse_grid, self.morphology_close(mask.unsqueeze(0)).squeeze(0)

    def normalize_grid(self, grid):
        h, w, _ = grid.shape  # grid dimensions
        grid = grid / torch.tensor([h, w]) * 2.0 - 1.0
        return grid

    def dilate(self, x, kernel_size = 3):
        kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32)

        # Apply convolution to simulate dilation
        # We use padding=1 to ensure the output size is the same as the input size
        output = F.conv2d(x, kernel, padding=1)

        # Apply a threshold to get a binary output
        dilated_image = (output > 0).float()

        return dilated_image
    
    def erode(self, x, kernel_size = 3):
        # Step 2: Erosion
        # For erosion, invert the image and kernel, apply dilation, then invert the output
        x = 1 - x
        inverted_kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32) # Same kernel as for dilation

        # Apply convolution (dilation on inverted image) with padding to maintain size
        eroded_output_inverted = F.conv2d(x, inverted_kernel, padding=1)

        # Invert the result to get the final eroded (closing) result
        eroded_image = 1 - (eroded_output_inverted > 0).float()

        return eroded_image

    def morphology_close(self, x, kernel_size=3):
        return self.erode(self.dilate(x, kernel_size), kernel_size)

    def add_speckle_noise(self, x, noise_variance=0.1):
        """
        Adds speckle noise to an image.

        Parameters:
        - x: A PyTorch tensor representing the image, shape (C, H, W) or (B, C, H, W)
        - noise_variance: Variance of the Gaussian noise

        Returns:
        - Noisy image: A PyTorch tensor of the same shape as `image` with speckle noise added
        """
        # Ensure the noise is generated for each image in the batch
        if x.dim() == 3: # For single image (C, H, W)
            noise_shape = x.shape
        elif x.dim() == 4: # For batch of images (B, C, H, W)
            noise_shape = (x.size(0), 1, x.size(2), x.size(3))

        # Generate noise from a normal distribution centered at 1 with specified variance
        noise = 1 + torch.randn(noise_shape, device=x.device) * noise_variance

        # Add noise to the image
        return x * noise

    def forward(self, x, grid=None, inverse_grid=None, mask_fan=None):

        if grid is None:

            #init tissue maps
            #generate maps from the dictionary and the input label map
            repeats = [1,]*len(x.shape)
            repeats[0] = x.shape[0]

            grid = self.grid
            inverse_grid = self.inverse_grid
            mask_fan = self.mask_fan

            grid = grid.repeat(repeats)
            inverse_grid = inverse_grid.repeat(repeats)
            mask_fan = mask_fan.repeat(repeats)
        
        x = x.to(torch.long)
        mean_diffusor = self.mean_diffusor_dict[x]
        variance_diffusor = self.variance_diffusor_dict[x]
        
        x = mean_diffusor + mean_diffusor * torch.randn(x.shape, device=x.device) * variance_diffusor
        x = torch.clip(x, min=0.0, max=1.0)

        return x*mask_fan
    
    
    
    def configure_optimizers(self):        
        
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        
        return optimizer
    

    def training_step(self, train_batch, batch_idx):
        seg = train_batch['seg']
        img = train_batch['img']

        fake_us = self(seg)[0]

        repeats = [1,]*len(img.shape)
        repeats[0] = img.shape[0]
        mask_fan = self.mask_fan.repeat(repeats)
        img = mask_fan*img

        loss = self.loss(fake_us, img)

        self.log("loss", loss)

        return loss
        
    
    def validation_step(self, val_batch, batch_idx):        
        seg = val_batch['seg']
        img = val_batch['img']

        fake_us = self(seg)[0]

        repeats = [1,]*len(img.shape)
        repeats[0] = img.shape[0]
        mask_fan = self.mask_fan.repeat(repeats)
        img = mask_fan*img

        val_loss = self.loss(fake_us, img)

        self.log("val_loss", val_loss, sync_dist=True)

    
