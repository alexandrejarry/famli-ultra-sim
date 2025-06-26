import torch
from torch import nn
import torch.nn.functional as F

from torchvision import transforms as T
from . import cut
from . import diffusion
from .lotus import UltrasoundRenderingLinear
import argparse
import numpy as np
import pandas as pd

class MergedCut3(nn.Module):
    def __init__(self):
        super().__init__()

        USR = torch.jit.load("/mnt/famli_netapp_shared/C1_ML_Analysis/src/famli-ultra-sim/trained_models/cut_v0.12-ae_v0.4_USR.pt")
        G = torch.jit.load("/mnt/famli_netapp_shared/C1_ML_Analysis/src/famli-ultra-sim/trained_models/cut_v0.12-ae_v0.4_G.pt")
        AE = torch.jit.load("/mnt/famli_netapp_shared/C1_ML_Analysis/src/famli-ultra-sim/trained_models/cut_v0.12-ae_v0.4_AE.pt")

        self.register_module('USR', USR)
        self.register_module('G', G)
        self.register_module('AE', AE)

        self.transform_us = T.Compose([T.Pad((0, 80, 0, 0)), T.CenterCrop(256)])

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

    def forward(self, X, grid, inverse_grid, mask_fan):
        
        X = self.USR(X, grid, inverse_grid, mask_fan)
        X = self.transform_us(X)
        X = self.G(X)
        return self.AE(X)[0]*self.transform_us(mask_fan)

class MergedLinearCut1Jit(nn.Module):
    def __init__(self):
        super().__init__()

        cut_fn = "/mnt/famli_netapp_shared/C1_ML_Analysis/src/famli-ultra-sim/trained_models/cutLinear_v1.0-ae_v0.4"
        USR = torch.jit.load(cut_fn + "_USR.pt")
        G = torch.jit.load(cut_fn + "_G.pt")
        AE = torch.jit.load("/mnt/famli_netapp_shared/C1_ML_Analysis/src/famli-ultra-sim/trained_models/cut_v0.12-ae_v0.4_AE.pt")

        self.register_module('USR', USR)
        self.register_module('G', G)
        self.register_module('AE', AE)

        self.transform_us = T.Compose([T.Pad((0, 80, 0, 0)), T.CenterCrop(256)])
        self.inverse_transform_us = T.Compose([T.Pad((0, 0, 0, 40)),  T.Lambda(lambda x: T.functional.crop(x, 40, 0, 256, 256))])

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

    def forward(self, X, grid, inverse_grid, mask_fan):
        
        X = self.USR(X, grid, inverse_grid, mask_fan)
        X = self.transform_us(X)
        X = self.G(X)
        return self.inverse_transform_us(self.AE(X)[0]*self.transform_us(mask_fan))
    

class MergedLinearCut1(nn.Module):
    def __init__(self):
        super().__init__()

        self.model_cut = cut.CutLinear.load_from_checkpoint("/mnt/raid/C1_ML_Analysis/train_output/ultra-sim/rendering_cut/v1.0/epoch=76-val_loss=3.16.ckpt")
        self.USR = self.model_cut.USR
        self.G = self.model_cut.G
        model_fn = "/mnt/raid/C1_ML_Analysis/train_output/diffusionAE/extract_frames_Dataset_C_masked_resampled_256_spc075_wscores_meta_BPD01_MACFL025-7mo-9mo/v0.4/epoch=72-val_loss=0.01.ckpt"
        self.AE = diffusion.AutoEncoderKL.load_from_checkpoint(model_fn)

        self.transform_us = T.Compose([T.Pad((0, 80, 0, 0)), T.CenterCrop(256)])
        self.inverse_transform_us = T.Compose([T.Pad((0, 0, 0, 40)),  T.Lambda(lambda x: T.functional.crop(x, 40, 0, 256, 256))])

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

    def forward(self, X, grid=None, inverse_grid=None, mask_fan=None):
        X = self.USR(X, grid, inverse_grid, mask_fan)
        X = self.transform_us(X)
        X = self.G(X)
        if mask_fan is None:
            mask_fan = self.USR.mask_fan
        return self.inverse_transform_us(self.AE(X)[0]*self.transform_us(mask_fan))
    


class MergedLinearCutLabel11(nn.Module):
    def __init__(self):
        super().__init__()

        self.model_cut = cut.CutLinear.load_from_checkpoint("/mnt/raid/C1_ML_Analysis/train_output/ultra-sim/rendering_cut_label11_linear/v0.2/epoch=74-val_loss=3.14.ckpt", num_labels=12)
        self.model_cut.freeze()
        self.USR = self.model_cut.USR
        self.G = self.model_cut.G
        # model_fn = "/mnt/raid/C1_ML_Analysis/train_output/diffusionAE/extract_frames_Dataset_C_masked_resampled_256_spc075_wscores_meta_BPD01_MACFL025-7mo-9mo/v0.4/epoch=72-val_loss=0.01.ckpt"
        # self.AE = diffusion.AutoEncoderKL.load_from_checkpoint(model_fn)

        self.transform_us = T.Compose([T.Pad((0, 80, 0, 0)), T.CenterCrop(256)])
        self.inverse_transform_us = T.Compose([T.Pad((0, 0, 0, 40)),  T.Lambda(lambda x: T.functional.crop(x, 40, 0, 256, 256))])

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

    def forward(self, X, grid=None, inverse_grid=None, mask_fan=None):
        X = self.USR(X, grid, inverse_grid, mask_fan)
        X = self.transform_us(X)
        X = self.G(X)
        X = self.inverse_transform_us(X)
        if mask_fan is None:
            mask_fan = self.USR.mask_fan
        # return self.inverse_transform_us(self.AE(X)[0]*self.transform_us(mask_fan))
        return X*mask_fan
    
class MergedCutLabel11(nn.Module):
    def __init__(self):
        super().__init__()

        self.model_cut = cut.CutLabel11.load_from_checkpoint("/mnt/raid/C1_ML_Analysis/train_output/ultra-sim/rendering_cut_label11/v0.1/epoch=85-val_loss=3.44.ckpt", num_labels=12)
        self.model_cut.freeze()

        self.USR = self.model_cut.USR
        self.G = self.model_cut.G
        # model_fn = "/mnt/raid/C1_ML_Analysis/train_output/diffusionAE/extract_frames_Dataset_C_masked_resampled_256_spc075_wscores_meta_BPD01_MACFL025-7mo-9mo/v0.4/epoch=72-val_loss=0.01.ckpt"
        # self.AE = diffusion.AutoEncoderKL.load_from_checkpoint(model_fn)

        self.transform_us = T.Compose([T.Pad((0, 80, 0, 0)), T.CenterCrop(256)])
        self.inverse_transform_us = T.Compose([T.Pad((0, 0, 0, 40)),  T.Lambda(lambda x: T.functional.crop(x, 40, 0, 256, 256))])

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

    def forward(self, X, grid=None, inverse_grid=None, mask_fan=None):
        X = self.USR(X, grid, inverse_grid, mask_fan)
        X = self.transform_us(X)
        X = self.G(X)
        X = self.inverse_transform_us(X)
        if mask_fan is None:
            mask_fan = self.USR.mask_fan
        # return self.inverse_transform_us(self.AE(X)[0]*self.transform_us(mask_fan))
        return X*mask_fan
    
class MergedUSRLabel11(nn.Module):
    def __init__(self):
        super().__init__()

        self.model_cut = cut.CutLabel11.load_from_checkpoint("/mnt/raid/C1_ML_Analysis/train_output/ultra-sim/rendering_cut_label11/v0.1/epoch=85-val_loss=3.44.ckpt", num_labels=12)
        self.model_cut.freeze()

        self.USR = self.model_cut.USR
        self.G = self.model_cut.G
        # model_fn = "/mnt/raid/C1_ML_Analysis/train_output/diffusionAE/extract_frames_Dataset_C_masked_resampled_256_spc075_wscores_meta_BPD01_MACFL025-7mo-9mo/v0.4/epoch=72-val_loss=0.01.ckpt"
        # self.AE = diffusion.AutoEncoderKL.load_from_checkpoint(model_fn)

        self.transform_us = T.Compose([T.Pad((0, 80, 0, 0)), T.CenterCrop(256)])
        self.inverse_transform_us = T.Compose([T.Pad((0, 0, 0, 40)),  T.Lambda(lambda x: T.functional.crop(x, 40, 0, 256, 256))])

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

    def forward(self, X, grid=None, inverse_grid=None, mask_fan=None):
        X = self.USR(X, grid, inverse_grid, mask_fan)
        # X = self.transform_us(X)
        # X = self.G(X)
        # X = self.inverse_transform_us(X)
        if mask_fan is None:
            mask_fan = self.USR.mask_fan
        # return self.inverse_transform_us(self.AE(X)[0]*self.transform_us(mask_fan))
        return X*mask_fan
    
class MergedLinearLabel11(nn.Module):
    def __init__(self):
        super().__init__()

        self.model_cut = cut.CutLinear.load_from_checkpoint("/mnt/raid/C1_ML_Analysis/train_output/ultra-sim/rendering_cut_label11_linear/v0.2/epoch=74-val_loss=3.14.ckpt", num_labels=12)
        self.model_cut.freeze()
        self.USR = self.model_cut.USR
        self.G = self.model_cut.G
        # model_fn = "/mnt/raid/C1_ML_Analysis/train_output/diffusionAE/extract_frames_Dataset_C_masked_resampled_256_spc075_wscores_meta_BPD01_MACFL025-7mo-9mo/v0.4/epoch=72-val_loss=0.01.ckpt"
        # self.AE = diffusion.AutoEncoderKL.load_from_checkpoint(model_fn)
        self.transform_us = T.Compose([T.Pad((0, 80, 0, 0)), T.CenterCrop(256)])
        self.inverse_transform_us = T.Compose([T.Pad((0, 0, 0, 40)),  T.Lambda(lambda x: T.functional.crop(x, 40, 0, 256, 256))])

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

    def forward(self, X, grid=None, inverse_grid=None, mask_fan=None, use_g=True):
        X = self.USR(X, grid, inverse_grid, mask_fan)

        if mask_fan is None:
            mask_fan = self.USR.mask_fan
        
        if use_g:
            X = self.transform_us(X)        
            X = self.G(X)
            X = self.inverse_transform_us(X)
        
        return X*mask_fan

class MergedLinearLabel11WOG(MergedLinearLabel11):
    def __init__(self):
        super().__init__()

    def forward(self, X, grid=None, inverse_grid=None, mask_fan=None):
        X = self.USR(X, grid, inverse_grid, mask_fan)

        if mask_fan is None:
            mask_fan = self.USR.mask_fan
            
        return X*mask_fan

class MergedLinearLabel11PassThrough(MergedLinearLabel11):
    def __init__(self):
        super().__init__()

    def forward(self, X, grid=None, inverse_grid=None, mask_fan=None):
        
        if mask_fan is None:
            mask_fan = self.USR.mask_fan
            
        return X*mask_fan

class MergedGuidedLabel11(MergedLinearLabel11):
    def __init__(self):
        super().__init__()
        self.au = torch.jit.load('/mnt/raid/C1_ML_Analysis/train_output/ultra-sim/guided/v0.1/model_traced.pt')

    def forward(self, X, grid=None, inverse_grid=None, mask_fan=None):
        # X = super().forward(X, grid, inverse_grid, mask_fan)
        if mask_fan is None:
            mask_fan = self.USR.mask_fan
        X, z_mu, z_sigma = self.au(X)
        return X*mask_fan

class MergedGuidedAnim(MergedLinearLabel11):
    def __init__(self):
        super().__init__()

        self.USR = UltrasoundRenderingLinear(num_labels=333, grid_w=256, grid_h=256, center_x=128.0, center_y=-30.0, r1=20.0, r2=215.0, theta=np.pi/4.0)
        df = pd.read_csv('/mnt/raid/C1_ML_Analysis/simulated_data_export/animation_export/shapes_intensity_map_nrrd.csv')
        self.USR.init_params(torch.tensor(df['mean']), torch.tensor(df['stddev']))

    def forward(self, X, grid=None, inverse_grid=None, mask_fan=None):
        X = self.USR(X, grid, inverse_grid, mask_fan)
        if mask_fan is None:
            mask_fan = self.USR.mask_fan
        return X*mask_fan

class CutLabel3D(MergedGuidedAnim):
    def __init__(self):
        super().__init__()
        self.model_cut = cut.CutLabel.load_from_checkpoint("/mnt/raid/C1_ML_Analysis/train_output/Cut3d/0.1", num_labels=12)
        self.model_cut.freeze()
        self.USR = self.model_cut.USR
        self.G = self.model_cut.G
        # model_fn = "/mnt/raid/C1_ML_Analysis/train_output/diffusionAE/extract_frames_Dataset_C_masked_resampled_256_spc075_wscores_meta_BPD01_MACFL025-7mo-9mo/v0.4/epoch=72-val_loss=0.01.ckpt"
        # self.AE = diffusion.AutoEncoderKL.load_from_checkpoint(model_fn)
