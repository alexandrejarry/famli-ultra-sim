import os

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
# from torch.cuda.amp import autocast, GradScaler

from generative.losses.adversarial_loss import PatchAdversarialLoss
from generative.losses.perceptual import PerceptualLoss
from generative.networks import nets
from generative.networks.nets.autoencoderkl import Encoder
from .diffusion import AutoEncoderKL



import numpy as np 
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from torchvision import transforms as T
from torchvision import models as tv_models
from torchvision.ops import Conv2dNormActivation

import monai
from monai.networks import nets as monai_nets
from monai.losses import DiceFocalLoss, DiceLoss, DiceCELoss
from monai.metrics import DiceMetric


import lightning as L
from lightning.pytorch.core import LightningModule

import pandas as pd

import pickle

import vtk
import math 

from .layers import TimeDistributed, MultiHeadAttention3D, ScoreLayer
from . import  us_simulation_jit

from typing import Tuple, Union

from monai.transforms import (            
    Compose,
    RandAffine,       
    RandAxisFlip, 
    RandRotate,    
    RandZoom,
)

from pytorch3d.structures import (
    Meshes,
    Pointclouds)

from pytorch3d.ops import (sample_points_from_meshes,
                           knn_points, 
                           knn_gather)

import sys
sys.path.append('/mnt/raid/C1_ML_Analysis/source/ShapeAXI/src/')

from shapeaxi.saxi_layers import AttentionChunk, SelfAttention, MHAContextModulated, ProjectionHead

from positional_encodings.torch_encodings import PositionalEncoding2D

from diffusers import DDPMScheduler
from diffusers.models import UNet2DConditionModel

from scipy.spatial.transform import Rotation as R

class VolumeSamplingBlindSweep(nn.Module):
    def __init__(self, mount_point="/mnt/raid/C1_ML_Analysis", simulation_fov_grid_size=[64, 128, 128], simulation_fov_fn='simulated_data_export/studies_merged/simulation_fov.stl', simulation_ultrasound_plane_fn='simulated_data_export/studies_merged/simulation_ultrasound_plane.stl', random_probe_pos_factor=0.0005, random_rotation_angles_ranges=((-10, 10), (-10, 10), (-10, 10))):
        super().__init__()

        # The simulation_fov is a cube that represent the bounds of the simulation. 
        reader = vtk.vtkSTLReader()
        reader.SetFileName(os.path.join(mount_point, simulation_fov_fn))
        reader.Update()
        simulation_fov = reader.GetOutput()
        simulation_fov_bounds = torch.tensor(simulation_fov.GetBounds())
        self.register_buffer('simulation_fov_bounds', simulation_fov_bounds)

        simulation_fov_origin = torch.tensor(simulation_fov_bounds[[0,2,4]])
        self.register_buffer('simulation_fov_origin', simulation_fov_origin)

        simulation_fov_end = torch.tensor(simulation_fov_bounds[[1,3,5]])
        self.register_buffer('simulation_fov_end', simulation_fov_end)

        # The simulation_fov_grid_size is the size of the grid that we will use to sample the simulation_fov
        simulation_fov_grid_size = torch.tensor(simulation_fov_grid_size)
        self.register_buffer('simulation_fov_grid_size', simulation_fov_grid_size)

        # The simulation_ultrasound_plane is the plane where the ultrasound simulation will be performed        
        if os.path.splitext(simulation_ultrasound_plane_fn)[1] == ".stl":
            reader = vtk.vtkSTLReader()
            reader.SetFileName(os.path.join(mount_point, simulation_ultrasound_plane_fn))
            reader.Update()
            simulation_ultrasound_plane = reader.GetOutput()
        elif os.path.splitext(simulation_ultrasound_plane_fn)[1] == ".obj":
            reader = vtk.vtkOBJReader()
            reader.SetFileName(os.path.join(mount_point, simulation_ultrasound_plane_fn))
            reader.Update()
            simulation_ultrasound_plane = reader.GetOutput()
        self.simulation_ultrasound_plane_bounds = np.array(simulation_ultrasound_plane.GetBounds())        
        
        # This is the ultrasound plane or the output image size of the ultrasound simulation
        simulation_ultrasound_plane_mesh_grid_size = [256, 256, 1]
        simulation_ultrasound_plane_mesh_grid_params = [torch.arange(start=start, end=end, step=(end - start)/simulation_ultrasound_plane_mesh_grid_size[idx]) for idx, (start, end) in enumerate(zip(self.simulation_ultrasound_plane_bounds[[0,2,4]], self.simulation_ultrasound_plane_bounds[[1,3,5]]))]
        simulation_ultrasound_plane_mesh_grid = torch.stack(torch.meshgrid(simulation_ultrasound_plane_mesh_grid_params, indexing='ij'), dim=-1).squeeze().to(torch.float32)

        self.register_buffer('simulation_ultrasound_plane_mesh_grid', simulation_ultrasound_plane_mesh_grid)

        self.random_probe_pos_factor = random_probe_pos_factor
        self.random_rotation_angles_ranges = random_rotation_angles_ranges
        
    def init_probe_params(self, mount_point="/mnt/raid/C1_ML_Analysis", probe_params_csv='simulated_data_export/studies_merged/probe_params.csv'):
        self.probe_params_df = pd.read_csv(os.path.join(mount_point, probe_params_csv))

        self.tags = self.probe_params_df['tag'].unique()
        self.tags_dict = {tag: idx for idx, tag in enumerate(self.tags)}
        grouped_df = self.probe_params_df.groupby('tag')

        # Load the probe parameters for the different sweeps. The paths of the sweeps are predefined
        for tag, group in grouped_df:
            sorted_group = group.sort_values('idx')

            probe_directions = []
            probe_origins = []

            for idx, row in sorted_group.iterrows():
                probe_params = pickle.load(open(os.path.join(mount_point, row['probe_param_fn']), 'rb'))
        
                probe_direction = torch.tensor(probe_params['probe_direction'], dtype=torch.float32, requires_grad=False)
                probe_origin = torch.tensor(probe_params['probe_origin'], dtype=torch.float32, requires_grad=False)

                probe_directions.append(probe_direction.T.unsqueeze(0))
                probe_origins.append(probe_origin.unsqueeze(0))

            self.register_buffer(f'probe_directions_{tag}', torch.cat(probe_directions, dim=0))
            self.register_buffer(f'probe_origins_{tag}', torch.cat(probe_origins, dim=0))

    def init_probe_params_from_pos(self, probe_paths, tags=["M", "L0", "L1", "R0", "R1", "C1", "C2", "C3", "C4"]):
        
        self.tags = tags

        self.tags_dict = {tag: idx for idx, tag in enumerate(self.tags)}

        for tag in self.tags:
            probe_origins = np.load(os.path.join(probe_paths, tag + ".npy"))

            self.register_buffer(f'probe_origins_{tag}', torch.tensor(probe_origins, dtype=torch.float32, requires_grad=False))

            probe_directions = np.load(os.path.join(probe_paths, tag + "_rotations.npy"))
            self.register_buffer(f'probe_directions_{tag}', torch.tensor(probe_directions, dtype=torch.float32, requires_grad=False))
            

    def transform_simulation_ultrasound_plane_tag(self, tag, probe_origin_rand=None, probe_direction_rand=None, use_random=False):
        # Given a sweep tag, we get ALL the planes in that sweep. We can use a random displacement/rotation for the probe. The use_random flag is used to determine if we want to use random displacements/rotations
        
        if torch.is_tensor(tag):
            tag = tag.item()
            tag = self.tags[tag]
        probe_directions = getattr(self, f'probe_directions_{tag}')
        probe_origins = getattr(self, f'probe_origins_{tag}')

        simulation_ultrasound_plane_mesh_grid_transformed_t = self.transform_simulation_ultrasound_plane(probe_directions, probe_origins, probe_origin_rand=probe_origin_rand, probe_direction_rand=probe_direction_rand, use_random=use_random)

        return simulation_ultrasound_plane_mesh_grid_transformed_t
    
    def transform_simulation_ultrasound_plane_norm(self, simulation_ultrasound_plane_mesh_grid_transformed, diffusor_origin, diffusor_end):
        # According to the documentation of F.grid_sample -> https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html, 
        # the values of the sampling grid must be between -1 and 1, i.e., the diffusor origin and end are [-1,-1,-1] and [1,1,1] respectively
        # We need transform the current set of transformed planes "simulation_ultrasound_plane_mesh_grid_transformed" to that space.
        diffusor_origin = diffusor_origin.view(diffusor_origin.shape[0], 1, 1, 1, 3)
        diffusor_end = diffusor_end.view(diffusor_end.shape[0], 1, 1, 1, 3)
        return 2.*(simulation_ultrasound_plane_mesh_grid_transformed - diffusor_origin)/(diffusor_end - diffusor_origin) - 1.

    def transform_simulation_ultrasound_plane(self, probe_directions, probe_origins, probe_origin_rand=None, probe_direction_rand=None, use_random=False):
        # Given probe directions and origins, we transform the default simulation ultrasound plane to those locations and directions
        simulation_ultrasound_plane_mesh_grid_transformed_t = []
        for probe_origin, probe_direction in zip(probe_origins, probe_directions): # We iterate through the BATCH otherwise this will not work for training
            simulation_ultrasound_plane_mesh_grid_transformed = self.transform_simulation_ultrasound_plane_single(probe_direction, probe_origin, probe_origin_rand=probe_origin_rand, probe_direction_rand=probe_direction_rand, use_random=use_random)
            simulation_ultrasound_plane_mesh_grid_transformed_t.append(simulation_ultrasound_plane_mesh_grid_transformed.unsqueeze(0))
        
        return torch.cat(simulation_ultrasound_plane_mesh_grid_transformed_t, dim=0)
    
    def transform_simulation_ultrasound_plane_single(self, probe_direction, probe_origin, probe_origin_rand=None, probe_direction_rand=None, use_random=False):
        # In a typical sweep we have 150+ probe_directions/_origin
        # Given a probe directions and origins, we transform the simulation ultrasound plane to those locations/directions
        # The ouptut is a locations in 3D space where the ultrasound simulation will be performed. We can use this to sample the diffusor at those locations
        if probe_origin_rand is not None:
            probe_origin = probe_origin + probe_origin_rand
        if probe_direction_rand is not None:
            probe_direction = torch.matmul(probe_direction, probe_direction_rand)

        if use_random:
            probe_origin_single_rand = torch.rand(3, device=probe_origin.device)*self.random_probe_pos_factor            
            probe_direction_single_rand = self.random_affine_matrix(self.random_rotation_angles_ranges).to(probe_direction.device)[0:3,0:3]            

            probe_origin = probe_origin + probe_origin_single_rand
            probe_direction = torch.matmul(probe_direction, probe_direction_single_rand)

        simulation_ultrasound_plane_mesh_grid_transformed = torch.matmul(self.simulation_ultrasound_plane_mesh_grid, probe_direction.T) + probe_origin
        return simulation_ultrasound_plane_mesh_grid_transformed.to(torch.float32)
    
    def diffusor_sampling_tag(self, tag, diffusor_t, diffusor_origin, diffusor_end, probe_origin_rand=None, probe_direction_rand=None, use_random=False):
        # Given a sweep tag, we sample the diffusor at the simulation ultrasound plane locations. The diffusor_origin and end are used to place this volume in space. 
        simulation_ultrasound_plane_mesh_grid_transformed_t = self.transform_simulation_ultrasound_plane_tag(tag, probe_origin_rand=probe_origin_rand, probe_direction_rand=probe_direction_rand, use_random=use_random)
        
        batch_size = diffusor_t.shape[0]
        simulation_ultrasound_plane_mesh_grid_transformed_norm_t = simulation_ultrasound_plane_mesh_grid_transformed_t.repeat(batch_size, 1, 1, 1, 1)
        simulation_ultrasound_plane_mesh_grid_transformed_norm_t = self.transform_simulation_ultrasound_plane_norm(simulation_ultrasound_plane_mesh_grid_transformed_norm_t, diffusor_origin, diffusor_end)
        
        return self.diffusor_sampling(diffusor_t, simulation_ultrasound_plane_mesh_grid_transformed_norm_t)
    
    def diffusor_sampling(self, diffusor_t, simulation_ultrasound_plane_mesh_grid_transformed_t):
        # sample the diffusor at the simulation ultrasound plane locations
        return F.grid_sample(diffusor_t, simulation_ultrasound_plane_mesh_grid_transformed_t, mode='nearest', align_corners=False)
    
    def diffusor_in_fov(self, diffusor_t, diffusor_origin, diffusor_end):
        # transforms the diffusor_t to the simulation fov. This is a resampling operation
        simulation_fov_origin = self.simulation_fov_bounds[[0,2,4]].flip(dims=[0])
        simulation_fov_end = self.simulation_fov_bounds[[1,3,5]].flip(dims=[0])
        
        simulation_fov_mesh_grid_params = [torch.arange(start=start, end=end, step=(end - start)/self.simulation_fov_grid_size[idx], device=self.simulation_fov_bounds.device) for idx, (start, end) in enumerate(zip(simulation_fov_origin, simulation_fov_end))]
        simulation_fov_mesh_grid = torch.stack(torch.meshgrid(simulation_fov_mesh_grid_params, indexing='ij'), dim=-1).squeeze().to(torch.float32)
        

        simulation_fov_mesh_grid_transformed = simulation_fov_mesh_grid.unsqueeze(0)
        repeats = [1,]*len(simulation_fov_mesh_grid_transformed.shape)
        repeats[0] = diffusor_t.shape[0]

        simulation_fov_mesh_grid_transformed = simulation_fov_mesh_grid_transformed.repeat(repeats)

        diffusor_origin = diffusor_origin.view(diffusor_t.shape[0], 1, 1, 1, 3)
        diffusor_end = diffusor_end.view(diffusor_t.shape[0], 1, 1, 1, 3)
        simulation_fov_mesh_grid_transformed = 2.*(simulation_fov_mesh_grid_transformed - diffusor_origin)/(diffusor_end - diffusor_origin) - 1.

        return F.grid_sample(diffusor_t.permute(0, 1, 4, 3, 2), simulation_fov_mesh_grid_transformed.float(), mode='nearest', align_corners=False)

    def simulated_sweep_in_fov(self, tag, sampled_sweep_simu):
        # Transform the simulated sweep to the simulation FOV
        # Get the default probe directions and origins
        assert len(sampled_sweep_simu.shape) == 5

        sampled_sweep_simu_shape = sampled_sweep_simu.shape[-3:]
        
        simulation_ultrasound_plane_mesh_grid_transformed_t = self.transform_simulation_ultrasound_plane_tag(tag)
        
        if simulation_ultrasound_plane_mesh_grid_transformed_t.shape[:3] != sampled_sweep_simu_shape:

                mesh_grid_params = [torch.arange(start=-1.0, end=1.0, step=(2.0/s), device=sampled_sweep_simu.device) for s in simulation_ultrasound_plane_mesh_grid_transformed_t.shape[:3]]
                z, y, x = torch.meshgrid(mesh_grid_params, indexing='ij')
                mesh_grid = torch.stack([x, y, z], dim=-1).to(torch.float32).unsqueeze(0)

                repeats = [1,]*len(mesh_grid.shape)
                repeats[0] = sampled_sweep_simu.shape[0]

                mesh_grid = mesh_grid.repeat(repeats)

                sampled_sweep_simu = F.grid_sample(sampled_sweep_simu, mesh_grid, align_corners=True)
        
        simulation_ultrasound_plane_mesh_grid_transformed_t = simulation_ultrasound_plane_mesh_grid_transformed_t.flip(dims=[-1])
        # Get the origin and end of the simulation FOV
        simulation_fov_origin = self.simulation_fov_bounds[[0,2,4]].flip(dims=[0])
        simulation_fov_end = self.simulation_fov_bounds[[1,3,5]].flip(dims=[0])
        # Convert the transformed planes to an ijk mapping
        simulation_ultrasound_plane_mesh_grid_transformed_t_idx = torch.clip((simulation_ultrasound_plane_mesh_grid_transformed_t - simulation_fov_origin)/(simulation_fov_end - simulation_fov_origin), 0, 1)*(self.simulation_fov_grid_size - 1)
        simulation_ultrasound_plane_mesh_grid_transformed_t_idx = simulation_ultrasound_plane_mesh_grid_transformed_t_idx.to(torch.int)
        simulation_ultrasound_plane_mesh_grid_transformed_t_idx = simulation_ultrasound_plane_mesh_grid_transformed_t_idx.reshape(-1, 3)

        out_fovs = []
        for ss_t in sampled_sweep_simu: #iterate through the batch
            
            out_fov = torch.zeros(self.simulation_fov_grid_size.tolist()).to(torch.float).to(self.simulation_fov_grid_size.device)
            # Here the trick is that the simulation_ultrasound_plane_mesh_grid_transformed_t_idx has the same shape as the sampled_sweep_simu. Some idx will be repeated/overwritten. 
            out_fov[simulation_ultrasound_plane_mesh_grid_transformed_t_idx[:,0], simulation_ultrasound_plane_mesh_grid_transformed_t_idx[:,1], simulation_ultrasound_plane_mesh_grid_transformed_t_idx[:,2]] = ss_t.reshape(-1)
            out_fovs.append(out_fov.unsqueeze(0))
        out_fovs = torch.cat(out_fovs, dim=0)

        return out_fovs
    
    def fov_physical(self, simulation_fov_grid_size=None):

        if simulation_fov_grid_size is None:
            simulation_fov_grid_size = self.simulation_fov_grid_size

        simulation_fov_mesh_grid_params = [torch.arange(end=s, device=self.simulation_fov_bounds.device) for s in simulation_fov_grid_size]
        simulation_fov_mesh_grid_idx = torch.stack(torch.meshgrid(simulation_fov_mesh_grid_params, indexing='ij'), dim=-1).squeeze().to(torch.float32)

        simulation_fov_origin = self.simulation_fov_bounds[[0,2,4]]
        simulation_fov_end = self.simulation_fov_bounds[[1,3,5]]
        simulation_fov_size = simulation_fov_grid_size.flip(dims=[0])

        simulation_fov_spacing = (simulation_fov_end - simulation_fov_origin)/simulation_fov_size
        return simulation_fov_origin + simulation_fov_mesh_grid_idx*simulation_fov_spacing
    
    def transform_fov_norm(self, X_physical):
        simulation_fov_origin = self.simulation_fov_bounds[[0,2,4]]
        simulation_fov_end = self.simulation_fov_bounds[[1,3,5]]

        return 2.*(X_physical - simulation_fov_origin)/(simulation_fov_end - simulation_fov_origin) - 1.

    
    def diffusor_resample(self, diffusor_t, size=128):

        simulation_fov_mesh_grid_params = [torch.arange(start=-1.0, end=1.0, step=(2.0/size), device=diffusor_t.device) for _ in range(3)]

        simulation_fov_mesh_grid = torch.stack(torch.meshgrid(simulation_fov_mesh_grid_params, indexing='ij'), dim=-1).to(torch.float32).unsqueeze(0)

        repeats = [1,]*len(simulation_fov_mesh_grid.shape)
        repeats[0] = diffusor_t.shape[0]

        simulation_fov_mesh_grid = simulation_fov_mesh_grid.repeat(repeats)

        return F.grid_sample(diffusor_t.permute(0, 1, 4, 3, 2), simulation_fov_mesh_grid.float(), mode='nearest', align_corners=False)
    
    def diffusor_tag_resample(self, diffusor_t, tag):

        assert len(diffusor_t.shape) == 5

        size = diffusor_t.shape[2:5]

        mesh_grid_params = [torch.arange(start=-1.0, end=1.0, step=(2.0/size[i]), device=diffusor_t.device) for i in range(3)]
        mesh_grid = torch.stack(torch.meshgrid(mesh_grid_params, indexing='ij'), dim=-1).to(torch.float32).unsqueeze(0)

        repeats = [1,]*len(mesh_grid.shape)
        repeats[0] = len(tag)

        mesh_grid = mesh_grid.repeat(repeats)

        x_v = []
        for t in tag:
            x_v.append(self.transform_simulation_ultrasound_plane_tag(t).permute(3, 0, 1, 2))
        x_v = torch.stack(x_v)

        return F.grid_sample(x_v, mesh_grid.float(), mode='nearest', align_corners=False).flatten(2).permute(0, 2, 1)

    def get_rotation_matrix(self, angle, axis):
        
        if axis == 'x':
            return torch.tensor([
                [1, 0, 0, 0],
                [0, math.cos(angle), -math.sin(angle), 0],
                [0, math.sin(angle), math.cos(angle), 0],
                [0, 0, 0, 1]
            ])
        elif axis == 'y':
            return torch.tensor([
                [math.cos(angle), 0, math.sin(angle), 0],
                [0, 1, 0, 0],
                [-math.sin(angle), 0, math.cos(angle), 0],
                [0, 0, 0, 1]
            ])
        elif axis == 'z':
            return torch.tensor([
                [math.cos(angle), -math.sin(angle), 0, 0],
                [math.sin(angle), math.cos(angle), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
    
    def random_affine_matrix(self, rotation_ranges):
        """
        Generate a random affine transformation matrix with rotation ranges specified.

        Args:
            rotation_ranges (tuple of tuples): ((min_angle_x, max_angle_x), (min_angle_y, max_angle_y), (min_angle_z, max_angle_z))
                                            Angles should be in degrees.

        Returns:
            torch.Tensor: A 4x4 affine transformation matrix.
        """

        rotation_angles = [torch.FloatTensor(1).uniform_(*range).item() for range in rotation_ranges]

        Rx = self.get_rotation_matrix(math.radians(rotation_angles[0]), 'x')
        Ry = self.get_rotation_matrix(math.radians(rotation_angles[1]), 'y')
        Rz = self.get_rotation_matrix(math.radians(rotation_angles[2]), 'z')

        affine_matrix = torch.mm(torch.mm(Rz, Ry), Rx)

        return affine_matrix

    def random_rotate_3d_batch(self, x):
        """
        Randomly rotates a batch of 3D image tensors and returns the rotated images along with the rotation matrices.
        
        Args:
            x (torch.Tensor): A tensor of shape (B, C, D, H, W).
        
        Returns:
            rotated_images (torch.Tensor): Rotated 3D tensors of shape (B, C, D, H, W).
            rotation_matrices (torch.Tensor): Rotation matrices of shape (B, 3, 3).
        """
        assert len(x.shape) == 5, "Input tensor must be 5D (B, C, D, H, W)."

        B, C, D, H, W = x.shape

        # Generate random angles for each batch (B, 3)
        angles = torch.rand(B, 3) * 2 * torch.pi  # Random angles between 0 and 2*pi

        # Generate rotation matrices for each sample in the batch
        rotation_matrices = torch.stack([
            self.get_rotation_matrix(angles[i, 2], 'z') @ 
            self.get_rotation_matrix(angles[i, 1], 'y') @ 
            self.get_rotation_matrix(angles[i, 0], 'x')
            for i in range(B)
        ]).to(x.device)

        # Convert 4x4 rotation matrices to 3x4 affine matrices
        affine_matrices = rotation_matrices.to(x.device)[:,:3,:]

        # Generate affine grids for each batch
        grids = F.affine_grid(
            affine_matrices,
            size=x.size(),
            align_corners=False
        )

        # Apply rotations using grid sampling
        x_rotated = F.grid_sample(
            x,
            grids,
            mode='nearest',
            padding_mode='zeros',
            align_corners=False
        )

        return x_rotated, rotation_matrices

    def apply_batch_rotation(self, V, rotation_matrices):
        """
        Applies a batch of rotation matrices to a batch of 3D points.

        Args:
            V (torch.Tensor): A tensor of shape (BS, N, 3) containing 3D points.
            rotation_matrices (torch.Tensor): A tensor of shape (BS, 3, 3) containing rotation matrices.

        Returns:
            rotated_points (torch.Tensor): A tensor of shape (BS, N, 3) with rotated points.
        """
        assert V.shape[-1] == 3, "Points tensor must have the last dimension of size 3 (3D coordinates)."
        assert rotation_matrices.shape[-2:] == (3, 3), "Rotation matrices must have shape (BS, 3, 3)."
        assert V.shape[0] == rotation_matrices.shape[0], "Batch size of points and rotation matrices must match."

        # Apply the rotation matrix to each batch element
        return torch.matmul(V, rotation_matrices)  # Transpose for proper multiplication but here we don't transpose because the V tensor is ordered XYZ while the rotation matrix is ordered ZYX
        # rotated_points = torch.matmul(V, rotation_matrices.transpose(1, 2))  # Transpose for proper multiplication
        # return rotated_points
    
    def get_sweep(self, X, X_origin, X_end, tag, use_random=False, simulator=None, grid=None, inverse_grid=None, mask_fan=None, return_masked=False):        

        probe_origin_rand = None
        probe_direction_rand = None            

        if use_random:
            probe_origin_rand = torch.rand(3, device=X.device)*0.0001
            probe_origin_rand = probe_origin_rand
            rotation_ranges = ((-5, 5), (-5, 5), (-10, 10))  # ranges in degrees for x, y, and z rotations
            probe_direction_rand = self.random_affine_matrix(rotation_ranges).to(X.device)[0:3,0:3]                        
        
        sampled_sweep = self.diffusor_sampling_tag(tag, X.to(torch.float), X_origin.to(torch.float), X_end.to(torch.float), probe_origin_rand=probe_origin_rand, probe_direction_rand=probe_direction_rand, use_random=use_random)

        if simulator is not None:
            with torch.no_grad():

                sampled_sweep_simu = []

                for ss in sampled_sweep:
                    ss_chunk = []
                    for ss_ch in torch.chunk(ss, chunks=20, dim=1):
                        simulated = simulator(ss_ch.unsqueeze(dim=1), grid=grid, inverse_grid=inverse_grid, mask_fan=mask_fan)
                        simulated = simulator.module.transform_us(simulated)
                        ss_chunk.append(simulated)
                    sampled_sweep_simu.append(torch.cat(ss_chunk, dim=2))

                sampled_sweep_simu = torch.stack(sampled_sweep_simu).detach()

                if return_masked:
                    if mask_fan == None:
                        mask_fan = simulator.module.USR.mask_fan
                    sampled_sweep = simulator.module.transform_us(sampled_sweep*mask_fan)
                    return sampled_sweep_simu, sampled_sweep

                return sampled_sweep_simu
        
        return sampled_sweep

    def embed_sweep(self, tag, sampled_sweep_simu):
        """ Embed the sweep with simulation FOV coordiantes
            Args: 
                tag: The tag of the sweep
                sampled_sweep_simu: The sampled sweep simulation with shape B, C, D, H, W
        """
        assert len(sampled_sweep_simu.shape) == 5 

        sampled_sweep_simu_shape = sampled_sweep_simu.shape[-3:]
        
        simulation_ultrasound_plane_mesh_grid_transformed_t = self.transform_simulation_ultrasound_plane_tag(tag)
        simulation_ultrasound_plane_mesh_grid_transformed_t = simulation_ultrasound_plane_mesh_grid_transformed_t.permute(3, 0, 1, 2).unsqueeze(0)

        if simulation_ultrasound_plane_mesh_grid_transformed_t.shape[2:] != sampled_sweep_simu_shape:

            mesh_grid_params = [torch.arange(start=-1.0, end=1.0, step=(2.0/s), device=sampled_sweep_simu.device) for s in sampled_sweep_simu_shape]
            z, y, x = torch.meshgrid(mesh_grid_params, indexing='ij')
            mesh_grid = torch.stack([x, y, z], dim=-1).to(torch.float32).unsqueeze(0)
            simulation_ultrasound_plane_mesh_grid_transformed_t = F.grid_sample(simulation_ultrasound_plane_mesh_grid_transformed_t, mesh_grid, align_corners=True)
        
        repeats = [1,]*len(simulation_ultrasound_plane_mesh_grid_transformed_t.shape)
        repeats[0] = sampled_sweep_simu.shape[0]
        simulation_ultrasound_plane_mesh_grid_transformed_t = simulation_ultrasound_plane_mesh_grid_transformed_t.repeat(repeats)

        return torch.cat([sampled_sweep_simu, simulation_ultrasound_plane_mesh_grid_transformed_t], dim=1)

class USDDPMPC(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        if self.hparams.autoencoder_fn is not None:
            au = AutoEncoderKL.load_from_checkpoint(self.hparams.autoencoder_fn)
            au.freeze()
            au = au.autoencoderkl
        else:
            au = nets.AutoencoderKL(
                spatial_dims=2,
                in_channels=self.hparams.in_channels,
                out_channels=self.hparams.out_channels,
                num_channels=self.hparams.num_channels,
                latent_channels=self.hparams.latent_channels,
                num_res_blocks=self.hparams.num_res_blocks,
                norm_num_groups=self.hparams.norm_num_groups,
                attention_levels=(False, False, True),
            )
        self.encoder = TimeDistributed(au.encoder, time_dim=2)
        self.quant_conv_mu = TimeDistributed(au.quant_conv_mu, time_dim=2)
        self.quant_conv_log_sigma = TimeDistributed(au.quant_conv_log_sigma, time_dim=2)
        
        self.vs = VolumeSamplingBlindSweep()

        self.attn_chunk = AttentionChunk(input_dim=(self.hparams.latent_channels*64*64), hidden_dim=64, chunks=self.hparams.n_chunks)
        self.proj = ProjectionHead(input_dim=(self.hparams.latent_channels*64*64), hidden_dim=1280, output_dim=self.hparams.embed_dim)

        self.dropout = nn.Dropout(self.hparams.dropout)

        self.p_encoding = PositionalEncoding2D(self.hparams.embed_dim)

        self.diffnet = UNet2DConditionModel(
            in_channels=self.hparams.input_dim, 
            out_channels=self.hparams.output_dim, 
            cross_attention_dim=self.hparams.embed_dim, 
            flip_sin_to_cos=self.hparams.flip_sin_to_cos,
            time_embedding_type=self.hparams.time_embedding_type,
            freq_shift=self.hparams.freq_shift)
        

        self.loss_fn = nn.MSELoss()

        self.noise_scheduler = DDPMScheduler(num_train_timesteps=self.hparams.num_train_steps, 
                                             beta_schedule="linear",
                                             clip_sample=False,
                                             prediction_type='epsilon')

        self.simu0 = TimeDistributed(us_simulation_jit.MergedLinearCutLabel11(), time_dim=2)
        self.simu1 = TimeDistributed(us_simulation_jit.MergedCutLabel11(), time_dim=2)
        self.simu2 = TimeDistributed(us_simulation_jit.MergedUSRLabel11(), time_dim=2)
        self.simu3 = TimeDistributed(us_simulation_jit.MergedLinearLabel11(), time_dim=2)
        self.simu4 = TimeDistributed(us_simulation_jit.MergedLinearLabel11WOG(), time_dim=2)

        
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("USDDPMPC")

        group.add_argument("--lr", type=float, default=1e-4)
        group.add_argument('--weight_decay', help='Weight decay for optimizer', type=float, default=0.01)
        
        # Image Encoder parameters 

        group.add_argument("--autoencoder_fn", type=str, default="/mnt/raid/C1_ML_Analysis/train_output/diffusionAE/extract_frames_Dataset_C_masked_resampled_256_spc075_wscores_meta_BPD01_MACFL025-7mo-9mo/v0.4/epoch=72-val_loss=0.01.ckpt", help='Pre trained autoencoder model')
        group.add_argument("--latent_channels", type=int, default=3, help='Output dimension for the image encoder')
        group.add_argument("--spatial_dims", type=int, default=2, help='Spatial dimensions for the autoencoder')
        group.add_argument("--in_channels", type=int, default=1, help='Input channels for the autoencoder')
        group.add_argument("--out_channels", type=int, default=1, help='Output channels for the autoencoder')
        group.add_argument("--num_channels", type=int, nargs="*", default=[128, 256, 384], help='Number of channels for each stage of the image encoder')
        group.add_argument("--num_res_blocks", type=int, default=1, help='Number of residual blocks')
        group.add_argument("--norm_num_groups", type=int, default=32, help='Number of groups for the normalization layer')
        group.add_argument("--n_chunks_e", type=int, default=10, help='Number of chunks in the encoder stage to reduce memory usage')
        group.add_argument("--n_chunks", type=int, default=8, help='Number of outputs in the time dimension')
        group.add_argument("--z_prob", type=float, default=0.2, help='Probability of dropping the latent code')
        
        
        # Encoder parameters for the diffusion model
        group.add_argument("--num_samples", type=int, default=4096, help='Number of samples to take from the mesh to start the encoding')
        group.add_argument("--input_dim", type=int, default=3, help='Input dimension for the encoder')
        group.add_argument("--embed_dim", type=int, default=1280, help='Embedding dimension')
        group.add_argument("--output_dim", type=int, default=3, help='Output dimension of the model')
        group.add_argument("--dropout", type=float, default=0.25, help='Dropout rate')
        
        group.add_argument("--num_train_steps", type=int, default=1000, help='Number of training steps for the noise scheduler')
        group.add_argument("--time_embedding_type", type=str, default='positional', help='Time embedding type', choices=['fourier', 'positional'])
        
        group.add_argument("--flip_sin_to_cos", type=int, default=1, help='Whether to flip sin to cos for Fourier time embedding.')
        group.add_argument("--freq_shift", type=int, default=0, help='Frequency shift for Fourier time embedding.')

        group.add_argument("--num_random_sweeps", type=int, default=0, help='How many random sweeps to use. 0 == all')
        group.add_argument("--n_grids", type=int, default=200, help='Number of grids')
        # group.add_argument("--target_label", type=int, default=7, help='Target label')

        # group.add_argument("--n_fixed_samples", type=int, default=12288, help='Number of fixed samples')

        return parent_parser
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        return optimizer

    def on_fit_start(self):        

        # Define the file names directly without using out_dir
        grid_t_file = 'grid_t.pt'
        inverse_grid_t_file = 'inverse_grid_t.pt'
        mask_fan_t_file = 'mask_fan_t.pt'

        if not os.path.exists(grid_t_file):
            grid_tensor = []
            inverse_grid_t = []
            mask_fan_t = []

            for i in range(self.hparams.n_grids):

                grid_w, grid_h = self.hparams.grid_w, self.hparams.grid_h
                center_x = self.hparams.center_x
                r1 = self.hparams.r1

                center_y = self.hparams.center_y_start + (self.hparams.center_y_end - self.hparams.center_y_start) * (torch.rand(1))
                r2 = self.hparams.r2_start + ((self.hparams.r2_end - self.hparams.r2_start) * torch.rand(1)).item()
                theta = self.hparams.theta_start + ((self.hparams.theta_end - self.hparams.theta_start) * torch.rand(1)).item()
                
                grid, inverse_grid, mask = self.USR.init_grids(grid_w, grid_h, center_x, center_y, r1, r2, theta)

                grid_tensor.append(grid.unsqueeze(dim=0))
                inverse_grid_t.append(inverse_grid.unsqueeze(dim=0))
                mask_fan_t.append(mask.unsqueeze(dim=0))

            self.grid_t = torch.cat(grid_tensor).to(self.device)
            self.inverse_grid_t = torch.cat(inverse_grid_t).to(self.device)
            self.mask_fan_t = torch.cat(mask_fan_t).to(self.device)

            # Save tensors directly to the current directory
            
            torch.save(self.grid_t, grid_t_file)
            torch.save(self.inverse_grid_t, inverse_grid_t_file)
            torch.save(self.mask_fan_t, mask_fan_t_file)

            # print("Grids SAVED!")
            # print(self.grid_t.shape, self.inverse_grid_t.shape, self.mask_fan_t.shape)
        
        elif not hasattr(self, 'grid_t'):
            # Load tensors directly from the current directory
            self.grid_t = torch.load(grid_t_file).to(self.device)
            self.inverse_grid_t = torch.load(inverse_grid_t_file).to(self.device)
            self.mask_fan_t = torch.load(mask_fan_t_file).to(self.device)

    def sampling(self, z_mu: torch.Tensor, z_sigma: torch.Tensor) -> torch.Tensor:        
        eps = torch.randn_like(z_sigma)
        z_vae = z_mu + eps * z_sigma
        return z_vae
    
    def compute_loss(self, X, X_hat, step="train", sync_dist=False):

        # Negative ELBO of P(X|z)
        loss = self.loss_fn(X, X_hat)
        
        self.log(f"{step}_loss", loss, sync_dist=sync_dist)

        return loss

    def volume_sampling(self, X, X_origin, X_end, use_random=False):
        with torch.no_grad():
            simulator = self.simu0
            
            grid = None
            inverse_grid = None
            mask_fan = None

            tags = self.vs.tags

            if use_random:

                simulator_idx = np.random.choice([0, 1, 2, 3, 4], replace=False)
                simulator = getattr(self, f'simu{simulator_idx}')

                if self.hparams.num_random_sweeps > 0:
                    tags = np.random.choice(self.vs.tags, self.hparams.num_random_sweeps)
                else:
                    tags = self.vs.tags

                grid_idx = torch.randint(low=0, high=self.hparams.n_grids - 1, size=(1,))
            
                grid = self.grid_t[grid_idx]
                inverse_grid = self.inverse_grid_t[grid_idx]
                mask_fan = self.mask_fan_t[grid_idx] 

            X_sweeps = []
            X_sweeps_tags = []            

            for tag in tags:                
                
                sampled_sweep_simu = self.vs.get_sweep(X, X_origin, X_end, tag, use_random=use_random, simulator=simulator, grid=grid, inverse_grid=inverse_grid, mask_fan=mask_fan)
                X_sweeps.append(sampled_sweep_simu)
                X_sweeps_tags.append(self.vs.tags_dict[tag])
                
            X_sweeps = torch.cat(X_sweeps, dim=1)
            X_sweeps_tags = torch.tensor(X_sweeps_tags).repeat(X_sweeps.shape[0], 1)

            return X_sweeps, X_sweeps_tags

    def positional_encoding(self, z, sweeps_tags):
        # Positional encoding for training only. Ensure the encoding is done in the correct order depending on the sweep
        # Define shapes
        BS, N, C, F = z.shape # [BS, N, self.hparams.n_chunks, 64*64*self.hparams.latent_channels]
        max_N = len(self.vs.tags)
        
        buffer = torch.zeros((BS, max_N, C, F), dtype=torch.float32, device=self.device)  # Buffer for data

        # Scatter data into the buffer
        for b in range(BS):
            buffer[b, sweeps_tags[b]] = z[b]

        buffer = self.p_encoding(buffer)
        
        # Assign output in tensor
        for b in range(BS):
            z[b] = buffer[b, sweeps_tags[b]]

        return z

    def training_step(self, train_batch, batch_idx):
        X, X_origin, X_end, X_PC = train_batch

        X, rotation_matrices = self.vs.random_rotate_3d_batch(X)

        batch_size = X.shape[0]

        if  torch.rand(1).item() < self.hparams.z_prob:
            z = torch.zeros(batch_size, 1, self.hparams.embed_dim, device=self.device)
        else:
            

            x_sweeps, sweeps_tags = self.volume_sampling(X, X_origin, X_end, use_random=True)

            # x_sweeps shape is B, N, C, T, H, W. N for number of sweeps ex. torch.Size([2, 2, 1, 200, 256, 256])
            # tags shape torch.Size([2, 2])
            Nsweeps = x_sweeps.shape[1] # Number of sweeps -> T
            
            z = []
            x_v = []

            for n in range(Nsweeps):
                x_sweeps_n = x_sweeps[:, n, :, :, :, :] # [BS, C, T, H, W]
                sweeps_tags_n = sweeps_tags[:, n]
                
                z_mu, z_sigma = self.encode(x_sweeps_n)            
                z_ = self.sampling(z_mu, z_sigma) 

                z_ = self.attn_chunk(z_) # [BS, self.hparams.latent_channels, self.hparams.n_chunks, 64. 64]

                z_ = z_.permute(0, 2, 3, 4, 1).reshape(batch_size, self.hparams.n_chunks, -1) # [BS, self.hparams.n_chunks, 64*64*self.hparams.latent_channels]

                z.append(z_.unsqueeze(1))

            z = torch.cat(z, dim=1) # [BS, N, self.hparams.n_chunks, 64*64*self.hparams.latent_channels]

            z = self.proj(z) # [BS, N, self.hparams.n_chunks, 1280]

            z = self.positional_encoding(z, sweeps_tags)
            z = z.view(batch_size, -1, self.hparams.embed_dim).contiguous()
            z = self.dropout(z)

        # Diffusion stage

        X_PC = self.vs.apply_batch_rotation(X_PC, rotation_matrices[:,0:3,0:3])

        noise = torch.randn_like(X_PC).to(self.device)

        timesteps = torch.randint(0, self.hparams.num_train_steps - 1, (X_PC.shape[0],)).long().to(self.device)

        noisy_X = self.noise_scheduler.add_noise(X_PC, noise=noise, timesteps=timesteps)

        X_hat = self(noisy_X.permute(0, 2, 1).view(-1, 3, 64, 64).contiguous(), timesteps=timesteps, context=z)
        X_hat = X_hat.view(-1, 3, 64*64).permute(0, 2, 1).contiguous()

        return self.compute_loss(X=noise, X_hat=X_hat)
        

    def validation_step(self, val_batch, batch_idx):
        
        X, X_origin, X_end, X_PC = val_batch

        x_sweeps, sweeps_tags = self.volume_sampling(X, X_origin, X_end)

        # x_sweeps shape is B, N, C, T, H, W. N for number of sweeps ex. torch.Size([2, 2, 1, 200, 256, 256]) 
        # tags shape torch.Size([2, 2])

        batch_size = x_sweeps.shape[0]
        Nsweeps = x_sweeps.shape[1] # Number of sweeps -> T
        
        z = []
        x_v = []

        for n in range(Nsweeps):
            x_sweeps_n = x_sweeps[:, n, :, :, :, :] # [BS, C, T, H, W]
            sweeps_tags_n = sweeps_tags[:, n]

            z_mu, z_sigma = self.encode(x_sweeps_n)
            z_ = z_mu

            z_ = self.attn_chunk(z_) # [BS, self.hparams.latent_channels, self.hparams.n_chunks, 64. 64]

            z_ = z_.permute(0, 2, 3, 4, 1).reshape(batch_size, self.hparams.n_chunks, -1) # [BS, self.hparams.n_chunks, 64*64*self.hparams.latent_channels]

            z.append(z_.unsqueeze(1))

        z = torch.cat(z, dim=1) # [BS, N, self.hparams.n_chunks, 64*64*self.hparams.latent_channels]

        z = self.proj(z) # [BS, N, elf.hparams.n_chunks, 1280]

        # We don't need to do the trick of using the buffer for the positional encoding here, ALL the sweeps are present in validation
        z = self.p_encoding(z)
        z = z.view(batch_size, -1, self.hparams.embed_dim).contiguous()

        # Diffusion stage
        noise = torch.randn_like(X_PC).to(self.device)

        timesteps = torch.randint(0, self.hparams.num_train_steps - 1, (X_PC.shape[0],)).long().to(self.device)

        noisy_X = self.noise_scheduler.add_noise(X_PC, noise=noise, timesteps=timesteps)

        X_hat = self(noisy_X.permute(0, 2, 1).view(-1, 3, 64, 64).contiguous(), timesteps=timesteps, context=z)
        X_hat = X_hat.view(-1, 3, 64*64).permute(0, 2, 1).contiguous()

        return self.compute_loss(X=noise, X_hat=X_hat, step="val", sync_dist=True)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        """
        Forwards an image through the spatial encoder, obtaining the latent mean and sigma representations.

        Args:
            x: BxCx[SPATIAL DIMS] tensor

        """
        h = []
        for x_chunk in x.chunk(self.hparams.n_chunks_e, dim=2):
            h.append(self.encoder(x_chunk))
        h = torch.cat(h, dim=2)

        z_mu = self.quant_conv_mu(h)
        z_log_var = self.quant_conv_log_sigma(h)
        z_log_var = torch.clamp(z_log_var, -30.0, 20.0)
        z_sigma = torch.exp(z_log_var / 2)

        return z_mu, z_sigma

    def forward(self, x: torch.tensor, timesteps: Union[torch.Tensor, float, int], context: torch.Tensor | None = None):

        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=self.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(self.device)

        if timesteps.shape[0] != x.shape[0]:
            timesteps = timesteps.repeat(x.shape[0])
            
        return self.diffnet(sample=x, timestep=timesteps, encoder_hidden_states=context).sample

    def sample(self, num_samples=1, intermediate_steps=None, z=None):
        intermediates = []

        X_t = torch.randn(num_samples, self.hparams.num_samples, self.hparams.input_dim).to(self.device)        

        for i, t in enumerate(self.noise_scheduler.timesteps):

            
            x_hat = self(X_t.permute(0, 2, 1).view(-1, self.hparams.input_dim, 64, 64).contiguous(), timesteps=t, context=z)  
            x_hat = x_hat.view(-1, self.hparams.input_dim, 64*64).permute(0, 2, 1).contiguous()

            # Update sample with step
            X_t = self.noise_scheduler.step(model_output=x_hat, timestep=t, sample=X_t).prev_sample
            # X_t = X_t.clone().detach()
            if intermediate_steps is not None and intermediate_steps > 0 and t % (self.hparams.num_train_steps//intermediate_steps) == 0:
                intermediates.append(X_t)

        return X_t, intermediates

    def sample_guided(self, num_samples=1, guidance_scale=7.5, intermediate_steps=None, z=None):
        intermediates = []

        # Initialize random noise
        device = self.device
        x_t = torch.randn(num_samples, 64*64, self.hparams.input_dim, device=device)

        for t in reversed(range(self.hparams.num_train_steps)):
            
            # Conditional prediction (with context)
            x_cond = self(
                x_t.permute(0, 2, 1).view(-1, self.hparams.input_dim, 64, 64).contiguous(),
                timesteps=t,
                context=z
            )
            x_cond = x_cond.view(-1, self.hparams.input_dim, 64*64).permute(0, 2, 1)

            # Unconditional prediction (without context)
            x_uncond = self(
                x_t.permute(0, 2, 1).view(-1, self.hparams.input_dim, 64, 64).contiguous(),
                timesteps=t,
                context=torch.zeros(num_samples, 1, self.hparams.embed_dim, device=device)
            )
            x_uncond = x_uncond.view(-1, self.hparams.input_dim, 64*64).permute(0, 2, 1).contiguous()

            # Perform classifier-free guidance
            x_guided = x_uncond + guidance_scale * (x_cond - x_uncond)

            # Update the diffusion step using guided output
            x_t = self.noise_scheduler.step(model_output=x_guided, t=t, sample=x_t)

            # Save intermediate steps if needed
            if intermediate_steps and t % (self.hparams.num_train_steps // intermediate_steps) == 0:
                intermediates.append(x_t)

        return x_t, intermediates

    # Given diffusion model: model(x, t, cond)
# x: noisy input at timestep t
# cond: conditional information, cond=None is unconditional

def guided_sampling(model, x_t, t, cond, guidance_scale):
    # Predict noise conditioned on cond
    eps_cond = model(x_t, t, cond)
    
    # Predict noise without conditioning (unconditional)
    eps_uncond = model(x_t, t, cond=None)
    
    # Perform classifier-free guidance
    eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
    
    return eps_guided


class USBabyFrame(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.vs = VolumeSamplingBlindSweep()
        
        encoder = monai.networks.nets.EfficientNetBN('efficientnet-b0', spatial_dims=2, in_channels=self.hparams.in_channels, num_classes=self.hparams.features)
        self.encoder = TimeDistributed(encoder, time_dim=2)

        p_encoding = torch.stack([self.positional_encoding(self.hparams.time_steps, self.hparams.features, tag) for tag in self.vs.tags_dict.values()])        
        self.register_buffer("p_encoding", p_encoding)
        
        self.proj = ProjectionHead(input_dim=self.hparams.features, hidden_dim=self.hparams.features, output_dim=self.hparams.embed_dim, activation=nn.LeakyReLU)
        self.attn_chunk = AttentionChunk(input_dim=self.hparams.embed_dim, hidden_dim=64, chunks=self.hparams.n_chunks)

        self.dropout = nn.Dropout(self.hparams.dropout)
        
        # 
        # self.mha = nn.MultiheadAttention(embed_dim=self.hparams.embed_dim, num_heads=self.hparams.num_heads, dropout=self.hparams.dropout, bias=False, batch_first=True)
        self.mha = MHAContextModulated(embed_dim=self.hparams.embed_dim, num_heads=self.hparams.num_heads, output_dim=self.hparams.embed_dim, dropout=self.hparams.dropout)
        
        # MHAContextModulated(embed_dim=self.hparams.embed_dim, num_heads=self.hparams.num_heads, output_dim=self.hparams.embed_dim, dropout=self.hparams.dropout)
        self.attn = SelfAttention(input_dim=self.hparams.embed_dim, hidden_dim=64)
        self.proj_final = ProjectionHead(input_dim=self.hparams.embed_dim, hidden_dim=64, output_dim=9, activation=nn.Tanh)        

        self.simu0 = TimeDistributed(us_simulation_jit.MergedLinearCutLabel11(), time_dim=2)
        self.simu1 = TimeDistributed(us_simulation_jit.MergedCutLabel11(), time_dim=2)
        self.simu2 = TimeDistributed(us_simulation_jit.MergedUSRLabel11(), time_dim=2)
        self.simu3 = TimeDistributed(us_simulation_jit.MergedLinearLabel11(), time_dim=2)
        self.simu4 = TimeDistributed(us_simulation_jit.MergedLinearLabel11WOG(), time_dim=2)
        
        belly_idx = np.load(self.hparams.belly_idx)
        self.register_buffer("belly_idx", torch.tensor(belly_idx, dtype=torch.long))
        head_idx = np.load(self.hparams.head_idx)        
        self.register_buffer("head_idx", torch.tensor(head_idx, dtype=torch.long))
        side_idx = np.load(self.hparams.side_idx)
        self.register_buffer("side_idx", torch.tensor(side_idx, dtype=torch.long))

        self.loss_fn = nn.CosineSimilarity(dim=2)
        # self.loss_fn = nn.MSELoss()

        
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("Calculate baby frame of orientation")

        group.add_argument("--lr", type=float, default=1e-4)
        group.add_argument('--weight_decay', help='Weight decay for optimizer', type=float, default=0.01)
        
        # Image Encoder parameters 
        group.add_argument("--spatial_dims", type=int, default=2, help='Spatial dimensions for the encoder')
        group.add_argument("--in_channels", type=int, default=1, help='Input channels for encoder')
        group.add_argument("--features", type=int, default=1280, help='Number of output features for the encoder')        
        group.add_argument("--n_chunks_e", type=int, default=16, help='Number of chunks in the encoder stage to reduce memory usage')
        group.add_argument("--n_chunks", type=int, default=16, help='Number of outputs in the time dimension')
        group.add_argument("--num_heads", type=int, default=8, help='Number of heads for multi_head attention')

        # Encoder parameters for the diffusion model
        group.add_argument("--num_samples", type=int, default=4096, help='Number of samples to take from the mesh to start the encoding')
        group.add_argument("--input_dim", type=int, default=3, help='Input dimension for the encoder')
        group.add_argument("--embed_dim", type=int, default=128, help='Embedding dimension')
        group.add_argument("--output_dim", type=int, default=3, help='Output dimension of the model')
        group.add_argument("--dropout", type=float, default=0.25, help='Dropout rate')
        
        
        group.add_argument("--time_steps", type=int, default=96, help='Number of time steps in the sweep or sequence length')
        group.add_argument("--num_random_sweeps", type=int, default=3, help='How many random sweeps to use. 0 == all')
        group.add_argument("--n_grids", type=int, default=200, help='Number of grids')
        group.add_argument("--belly_idx", type=str, default='/mnt/raid/C1_ML_Analysis/simulated_data_export/fetus_rest_selected/belly_idx.npy', help='Indices for belly')
        group.add_argument("--head_idx", type=str, default='/mnt/raid/C1_ML_Analysis/simulated_data_export/fetus_rest_selected/head_idx.npy', help='Indices for head')
        group.add_argument("--side_idx", type=str, default='/mnt/raid/C1_ML_Analysis/simulated_data_export/fetus_rest_selected/side_idx.npy', help='Indices for side')
        
        
        # group.add_argument("--p_ids", type=int, nargs='+', default=[1470, 3369, 2043], help='Point ids to compute the orthogonal orientation frame')
        # group.add_argument("--target_label", type=int, default=7, help='Target label')

        # group.add_argument("--n_fixed_samples", type=int, default=12288, help='Number of fixed samples')

        return parent_parser
    
    def positional_encoding(self, seq_len: int, d_model: int, tag: int) -> torch.Tensor:
        """
        Sinusoidal positional encoding with tag-based offset.

        Args:
            seq_len (int): Sequence length.
            d_model (int): Embedding dimension.
            tag (int): Unique tag for the sequence.
            device (str): Device to store the tensor.

        Returns:
            torch.Tensor: Positional encoding (seq_len, d_model).
        """
        pe = torch.zeros(seq_len, d_model)
        
        # Offset positions by a tag-dependent amount to make each sequence encoding unique
        position = torch.arange(tag * seq_len, (tag + 1) * seq_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        return optimizer

    def on_fit_start(self):        

        # Define the file names directly without using out_dir
        grid_t_file = 'grid_t.pt'
        inverse_grid_t_file = 'inverse_grid_t.pt'
        mask_fan_t_file = 'mask_fan_t.pt'

        if not os.path.exists(grid_t_file):
            grid_tensor = []
            inverse_grid_t = []
            mask_fan_t = []

            for i in range(self.hparams.n_grids):

                grid_w, grid_h = self.hparams.grid_w, self.hparams.grid_h
                center_x = self.hparams.center_x
                r1 = self.hparams.r1

                center_y = self.hparams.center_y_start + (self.hparams.center_y_end - self.hparams.center_y_start) * (torch.rand(1))
                r2 = self.hparams.r2_start + ((self.hparams.r2_end - self.hparams.r2_start) * torch.rand(1)).item()
                theta = self.hparams.theta_start + ((self.hparams.theta_end - self.hparams.theta_start) * torch.rand(1)).item()
                
                grid, inverse_grid, mask = self.USR.init_grids(grid_w, grid_h, center_x, center_y, r1, r2, theta)

                grid_tensor.append(grid.unsqueeze(dim=0))
                inverse_grid_t.append(inverse_grid.unsqueeze(dim=0))
                mask_fan_t.append(mask.unsqueeze(dim=0))

            self.grid_t = torch.cat(grid_tensor).to(self.device)
            self.inverse_grid_t = torch.cat(inverse_grid_t).to(self.device)
            self.mask_fan_t = torch.cat(mask_fan_t).to(self.device)

            # Save tensors directly to the current directory
            
            torch.save(self.grid_t, grid_t_file)
            torch.save(self.inverse_grid_t, inverse_grid_t_file)
            torch.save(self.mask_fan_t, mask_fan_t_file)

            # print("Grids SAVED!")
            # print(self.grid_t.shape, self.inverse_grid_t.shape, self.mask_fan_t.shape)
        
        elif not hasattr(self, 'grid_t'):
            # Load tensors directly from the current directory
            self.grid_t = torch.load(grid_t_file).to(self.device)
            self.inverse_grid_t = torch.load(inverse_grid_t_file).to(self.device)
            self.mask_fan_t = torch.load(mask_fan_t_file).to(self.device)
    
    def compute_loss(self, Y, X_hat, step="train", sync_dist=False):
        
        loss = self.loss_fn(Y, X_hat) # The cosine similarity loss is close to 1 if the vectores are pointing in the same direction

        loss_mean = torch.mean(loss) # This should be close to 1
        # loss_std = torch.std(loss) # This should be close to 0        
        
        loss = torch.sum(torch.square(1.0 - loss)) # We minimize 1 - cos_sim so the vector are as close as possible to each other
        
        self.log(f"{step}_loss", loss, sync_dist=sync_dist)
        self.log(f"{step}_loss_mean", loss_mean, sync_dist=sync_dist)
        # self.log(f"{step}_loss_std", loss_std, sync_dist=sync_dist)

        return loss

    def volume_sampling(self, X, X_origin, X_end, use_random=False):
        with torch.no_grad():
            simulator = self.simu0
            
            grid = None
            inverse_grid = None
            mask_fan = None

            tags = self.vs.tags

            if use_random:

                simulator_idx = np.random.choice([0, 1, 2, 3, 4], replace=False)
                simulator = getattr(self, f'simu{simulator_idx}')
                # simulator = self.simu0

                if self.hparams.num_random_sweeps > 0:
                    tags = np.random.choice(self.vs.tags, self.hparams.num_random_sweeps)
                else:
                    tags = self.vs.tags

                grid_idx = torch.randint(low=0, high=self.hparams.n_grids - 1, size=(1,))
            
                grid = self.grid_t[grid_idx]
                inverse_grid = self.inverse_grid_t[grid_idx]
                mask_fan = self.mask_fan_t[grid_idx] 

            X_sweeps = []
            X_sweeps_tags = []            

            for tag in tags:                
                
                sampled_sweep_simu = self.vs.get_sweep(X, X_origin, X_end, tag, use_random=use_random, simulator=simulator, grid=grid, inverse_grid=inverse_grid, mask_fan=mask_fan)
                r_idx = torch.randint(low=0, high=sampled_sweep_simu.shape[3], size=(self.hparams.time_steps,))
                r_idx = torch.sort(r_idx)[0]                
                sampled_sweep_simu = sampled_sweep_simu[:, :, :, r_idx, :, :]                
                X_sweeps.append(sampled_sweep_simu)
                X_sweeps_tags.append(self.vs.tags_dict[tag])
                
            X_sweeps = torch.cat(X_sweeps, dim=1)
            X_sweeps_tags = torch.tensor(X_sweeps_tags, device=self.device)

            return X_sweeps, X_sweeps_tags
    
    def compute_orthogonal_frame(self, pc: torch.Tensor) -> torch.Tensor:    
        """
        Given point clouds,
        returns a tensor of shape [B, 3, 3] representing an orthogonal frame [x, y, z] for each batch.
        """

        head_idx = self.head_idx.expand(pc.shape[0], -1, -1)
        belly_idx = self.belly_idx.expand(pc.shape[0], -1, -1)
        side_idx = self.side_idx.expand(pc.shape[0], -1, -1)

        pc_head_k = knn_gather(pc, head_idx).squeeze(2)
        pc_belly_k = knn_gather(pc, belly_idx).squeeze(2)
        pc_side_k = knn_gather(pc, side_idx).squeeze(2)

        points = torch.stack([torch.mean(pc_belly_k, dim=1),
                            torch.mean(pc_head_k, dim=1),
                            torch.mean(pc_side_k, dim=1)], dim=1)

        p0 = points[:, 0]
        p1 = points[:, 1]
        p2 = points[:, 2]
        
        v1 = p1 - p0
        v2 = p2 - p0

        # Normalize x (first direction)
        x = F.normalize(v1, dim=1)

        # Compute z = normalized cross(v1, v2)
        z = F.normalize(torch.cross(v1, v2, dim=1), dim=1)

        # Compute y = cross(z, x)
        y = torch.cross(z, x, dim=1)

        # Stack the vectors as rows of the rotation matrix
        frame = torch.stack([x, y, z], dim=1)  # [B, 3, 3]

        return frame, points

    def training_step(self, train_batch, batch_idx):
        X, X_origin, X_end, X_PC = train_batch

        X, rotation_matrices = self.vs.random_rotate_3d_batch(X)

        x_sweeps, sweeps_tags = self.volume_sampling(X, X_origin, X_end, use_random=True)

        x_hat = self(x_sweeps, sweeps_tags)        

        X_PC = self.vs.apply_batch_rotation(X_PC, rotation_matrices[:,0:3,0:3])
        
        y, y_p = self.compute_orthogonal_frame(X_PC)        

        return self.compute_loss(Y=y, X_hat=x_hat, step="train")
        

    def validation_step(self, val_batch, batch_idx):
        
        X, X_origin, X_end, X_PC = val_batch

        x_sweeps, sweeps_tags = self.volume_sampling(X, X_origin, X_end)

        x_hat = self(x_sweeps, sweeps_tags)

        y, y_p = self.compute_orthogonal_frame(X_PC)        

        return self.compute_loss(Y=y, X_hat=x_hat, step="val", sync_dist=True)

        

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        """
        Forwards an image through the spatial encoder, obtaining the latent mean and sigma representations.

        Args:
            x: BxCx[SPATIAL DIMS] tensor

        """
        z = []
        for x_chunk in x.chunk(self.hparams.n_chunks_e, dim=2):
            z.append(self.encoder(x_chunk))
        z = torch.cat(z, dim=2)

        return z

    def forward(self, x_sweeps: torch.tensor, sweeps_tags: torch.tensor):
        
        batch_size = x_sweeps.shape[0]
        # x_sweeps shape is B, N, C, T, H, W. N for number of sweeps ex. torch.Size([2, 2, 1, 200, 256, 256]) 
        # tags shape torch.Size([2, 2])
        Nsweeps = x_sweeps.shape[1] # Number of sweeps -> T

        z = []
        x_v = []

        for n in range(Nsweeps):
            x_sweeps_n = x_sweeps[:, n, :, :, :, :] # [BS, C, T, H, W]
            
            tag = sweeps_tags[n]
            # x_sweeps_n = self.vs.embed_sweep(tag, x_sweeps_n)

            z_ = self.encode(x_sweeps_n) # [BS, T, self.hparams.features]

            # tag = sweeps_tags[n]
            p_enc = self.p_encoding[tag].unsqueeze(0)

            z_ = z_.permute(0, 2, 1) # Permute the time dim with the output features. -> Shape is now [BS, T, F]
            z_ = z_ + p_enc # [BS, T, self.hparams.features]            
            z_ = self.proj(z_) # [BS, self.hparams.n_chunks, self.hparams.embed_dim]

            z_ = self.attn_chunk(z_) # [BS, self.hparams.n_chunks, self.hparams.features]
            z_ = self.mha(z_)
            
            z.append(z_)

        z = torch.stack(z, dim=1) # [BS, N, self.hparams.n_chunks, 64*64*self.hparams.latent_channels]
        z = z.view(batch_size, -1, self.hparams.embed_dim).contiguous()

        z, z_s = self.attn(z, z)

        x_hat = self.proj_final(z)        
        x_hat = x_hat.view(batch_size, 3, 3).contiguous() # [BS, 3, 3]
        x_hat = F.normalize(x_hat, dim=2)

        return x_hat

    # Given diffusion model: model(x, t, cond)
# x: noisy input at timestep t
# cond: conditional information, cond=None is unconditional

def guided_sampling(model, x_t, t, cond, guidance_scale):
    # Predict noise conditioned on cond
    eps_cond = model(x_t, t, cond)
    
    # Predict noise without conditioning (unconditional)
    eps_uncond = model(x_t, t, cond=None)
    
    # Perform classifier-free guidance
    eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
    
    return eps_guided


class USSeg(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.vs = VolumeSamplingBlindSweep()

        self.model = monai_nets.UNet(
            spatial_dims=self.hparams.spatial_dims,
            in_channels=self.hparams.in_channels,
            out_channels=self.hparams.out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=self.hparams.norm,
        )
        self.loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
        # self.post_pred = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(argmax=True, to_onehot=2)])
        # self.post_label = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(to_onehot=2)])
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

        self.simu0 = TimeDistributed(us_simulation_jit.MergedLinearCutLabel11(), time_dim=2)
        self.simu1 = TimeDistributed(us_simulation_jit.MergedCutLabel11(), time_dim=2)
        self.simu2 = TimeDistributed(us_simulation_jit.MergedUSRLabel11(), time_dim=2)
        self.simu3 = TimeDistributed(us_simulation_jit.MergedLinearLabel11(), time_dim=2)
        self.simu4 = TimeDistributed(us_simulation_jit.MergedLinearLabel11WOG(), time_dim=2)
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("Calculate baby frame of orientation")

        group.add_argument("--lr", type=float, default=1e-4)
        group.add_argument('--weight_decay', help='Weight decay for optimizer', type=float, default=0.01)
        
        # Image Encoder parameters 
        group.add_argument("--spatial_dims", type=int, default=3, help='Spatial dimensions for the encoder')
        group.add_argument("--in_channels", type=int, default=1, help='Input number of channels')
        group.add_argument("--out_channels", type=int, default=12, help='Output number of channels')
        group.add_argument("--time_steps", type=int, default=128, help='Sample N number of frames from sweep')
        group.add_argument("--norm", type=str, default='BATCH', help='Type of norm')
        group.add_argument("--num_sweeps", type=int, default=1, help='Sample N sweeps from the volume')
        group.add_argument("--n_grids", type=int, default=200, help='Number of grids')
        
        
        # group.add_argument("--p_ids", type=int, nargs='+', default=[1470, 3369, 2043], help='Point ids to compute the orthogonal orientation frame')
        # group.add_argument("--target_label", type=int, default=7, help='Target label')

        # group.add_argument("--n_fixed_samples", type=int, default=12288, help='Number of fixed samples')

        return parent_parser
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        return optimizer

    def on_fit_start(self):

        # Define the file names directly without using out_dir
        grid_t_file = 'grid_t.pt'
        inverse_grid_t_file = 'inverse_grid_t.pt'
        mask_fan_t_file = 'mask_fan_t.pt'

        if not os.path.exists(grid_t_file):
            grid_tensor = []
            inverse_grid_t = []
            mask_fan_t = []

            for i in range(self.hparams.n_grids):

                grid_w, grid_h = self.hparams.grid_w, self.hparams.grid_h
                center_x = self.hparams.center_x
                r1 = self.hparams.r1

                center_y = self.hparams.center_y_start + (self.hparams.center_y_end - self.hparams.center_y_start) * (torch.rand(1))
                r2 = self.hparams.r2_start + ((self.hparams.r2_end - self.hparams.r2_start) * torch.rand(1)).item()
                theta = self.hparams.theta_start + ((self.hparams.theta_end - self.hparams.theta_start) * torch.rand(1)).item()
                
                grid, inverse_grid, mask = self.USR.init_grids(grid_w, grid_h, center_x, center_y, r1, r2, theta)

                grid_tensor.append(grid.unsqueeze(dim=0))
                inverse_grid_t.append(inverse_grid.unsqueeze(dim=0))
                mask_fan_t.append(mask.unsqueeze(dim=0))

            self.grid_t = torch.cat(grid_tensor).to(self.device)
            self.inverse_grid_t = torch.cat(inverse_grid_t).to(self.device)
            self.mask_fan_t = torch.cat(mask_fan_t).to(self.device)

            # Save tensors directly to the current directory
            
            torch.save(self.grid_t, grid_t_file)
            torch.save(self.inverse_grid_t, inverse_grid_t_file)
            torch.save(self.mask_fan_t, mask_fan_t_file)

            # print("Grids SAVED!")
            # print(self.grid_t.shape, self.inverse_grid_t.shape, self.mask_fan_t.shape)
        
        elif not hasattr(self, 'grid_t'):
            # Load tensors directly from the current directory
            self.grid_t = torch.load(grid_t_file).to(self.device)
            self.inverse_grid_t = torch.load(inverse_grid_t_file).to(self.device)
            self.mask_fan_t = torch.load(mask_fan_t_file).to(self.device)
    
    def compute_loss(self, Y, X_hat, step="train", sync_dist=False):
        
        loss = self.loss_fn(X_hat, Y) 
        
        self.log(f"{step}_loss", loss, sync_dist=sync_dist)
        # self.log(f"{step}_loss_mean", loss_mean, sync_dist=sync_dist)
        # self.log(f"{step}_loss_std", loss_std, sync_dist=sync_dist)

        return loss

    def volume_sampling(self, X, X_origin, X_end, use_random=False):
        with torch.no_grad():
            simulator = self.simu0
            
            grid = None
            inverse_grid = None
            mask_fan = None

            tags = self.vs.tags

            if use_random:

                simulator_idx = np.random.choice([0, 1, 2, 3, 4], replace=False)
                simulator = getattr(self, f'simu{simulator_idx}')

                grid_idx = torch.randint(low=0, high=self.hparams.n_grids - 1, size=(1,))
            
                grid = self.grid_t[grid_idx]
                inverse_grid = self.inverse_grid_t[grid_idx]
                mask_fan = self.mask_fan_t[grid_idx] 

            
            tags = np.random.choice(self.vs.tags, self.hparams.num_sweeps)
                

            X_sweeps = []
            Y_sweeps = []
            X_sweeps_tags = []            

            for tag in tags:                
                
                sampled_sweep_simu, sampled_sweep = self.vs.get_sweep(X, X_origin, X_end, tag, use_random=use_random, simulator=simulator, grid=grid, inverse_grid=inverse_grid, mask_fan=mask_fan, return_masked=True)
                
                r_idx = torch.randint(low=0, high=sampled_sweep_simu.shape[3], size=(self.hparams.time_steps,))
                r_idx = torch.sort(r_idx)[0]
                sampled_sweep_simu = sampled_sweep_simu[:, :, :, r_idx, :, :]
                sampled_sweep = sampled_sweep[:, :, r_idx, :, :]        
                X_sweeps.append(sampled_sweep_simu)
                X_sweeps_tags.append(self.vs.tags_dict[tag])

                Y_sweeps.append(sampled_sweep)
                
            X_sweeps = torch.cat(X_sweeps, dim=1)
            X_sweeps_tags = torch.tensor(X_sweeps_tags, device=self.device)
            Y_sweeps = torch.cat(Y_sweeps, dim=1)

            return X_sweeps, Y_sweeps, X_sweeps_tags

    def training_step(self, train_batch, batch_idx):
        X, X_origin, X_end, X_PC = train_batch

        X, rotation_matrices = self.vs.random_rotate_3d_batch(X)

        x, y, sweeps_tags = self.volume_sampling(X, X_origin, X_end, use_random=True)
        x = x.squeeze(1)

        x_hat = self(x)

        return self.compute_loss(Y=y, X_hat=x_hat, step="train")
        

    def validation_step(self, val_batch, batch_idx):
        
        X, X_origin, X_end, X_PC = val_batch

        x, y, sweeps_tags = self.volume_sampling(X, X_origin, X_end)
        x = x.squeeze(1)

        x_hat = self(x)

        self.dice_metric(y=y, y_pred=x_hat)

        return self.compute_loss(Y=y, X_hat=x_hat, step="val", sync_dist=True)

    def on_validation_epoch_end(self):
        # aggregate the final mean dice result
        metric = self.dice_metric.aggregate().item()

        self.log("val_dice", metric, sync_dist=True)
        
        self.dice_metric.reset()

    def forward(self, x_sweeps: torch.tensor):
        return self.model(x_sweeps)

# class USPCReconstruction(LightningModule):
#     def __init__(self, **kwargs):
#         super().__init__()
#         self.save_hyperparameters()

#         self.encoder_i = TimeDistributed(Encoder(spatial_dims=2,
#             in_channels=1,
#             num_channels=[8, 16, 32, 64, 128],
#             out_channels=self.hparams.latent_channels,
#             num_res_blocks=[2, 2, 2, 2, 2],
#             norm_num_groups=8,
#             norm_eps=1e-3,
#             attention_levels=[False, False, False, False, True],
#             with_nonlocal_attn=True), time_dim=2) # [BS, latent_channels, T, H, W]

#         self.attn_chunk = AttentionChunk(input_dim=(self.hparams.latent_channels*16*16), hidden_dim=64, chunks=16)

#         self.encoder_v = MHAIdxEncoder(input_dim=self.hparams.input_dim, 
#                                      output_dim=self.hparams.stages[-1], 
#                                      K=self.hparams.K, 
#                                      num_heads=self.hparams.num_heads, 
#                                      stages=self.hparams.stages, 
#                                      dropout=self.hparams.dropout, 
#                                      pooling_factor=self.hparams.pooling_factor, 
#                                      pooling_hidden_dim=self.hparams.pooling_hidden_dim,
#                                      score_pooling=self.hparams.score_pooling,
#                                      feed_forward_hidden_dim=self.hparams.feed_forward_hidden_dim, 
#                                     #  use_skip_connection=self.hparams.use_skip_connection, 
#                                      return_v=True)
        
#         self.fc = nn.Linear(self.hparams.stages[-1], self.hparams.output_dim, bias=False)
        
#         # self.ff_mu = FeedForward(self.hparams.stages[-1], hidden_dim=self.hparams.stages[-1], dropout=self.hparams.dropout)
#         # self.ff_sigma = FeedForward(self.hparams.stages[-1], hidden_dim=self.hparams.stages[-1], dropout=self.hparams.dropout)
        
#         # self.decoder_v = MHAIdxDecoder(input_dim=self.hparams.stages[-1], 
#         #                              output_dim=self.hparams.output_dim, 
#         #                              K=self.hparams.K[::-1], 
#         #                              num_heads=self.hparams.num_heads[::-1], 
#         #                              stages=self.hparams.stages[::-1], 
#         #                              dropout=self.hparams.dropout, 
#         #                              pooling_hidden_dim=self.hparams.pooling_hidden_dim[::-1] if self.hparams.pooling_hidden_dim is not None else None,
#         #                              feed_forward_hidden_dim=self.hparams.feed_forward_hidden_dim[::-1] if self.hparams.feed_forward_hidden_dim is not None else None,
#         #                              use_skip_connection=self.hparams.use_skip_connection)

        
#         self.vs = VolumeSamplingBlindSweep(mount_point=self.hparams.mount_point)

#         us_simulator_cut = MergedLinearLabel11().eval()
#         self.us_simulator_cut_td = TimeDistributed(us_simulator_cut, time_dim=2).eval()


#         if hasattr(self.hparams, 'n_fixed_samples') and self.hparams.n_fixed_samples is not None and self.hparams.n_fixed_samples > 0:
#             x_v_fixed = torch.rand(1, self.hparams.n_fixed_samples, 3)
#             x_v_fixed = (self.vs.simulation_fov_end - self.vs.simulation_fov_origin)*x_v_fixed + self.vs.simulation_fov_origin
#             self.register_buffer('x_v_fixed', x_v_fixed)
    
#     @staticmethod
#     def add_model_specific_args(parent_parser):
#         group = parent_parser.add_argument_group("USPCReconstruction")

#         group.add_argument("--lr", type=float, default=1e-4)
#         group.add_argument('--weight_decay', help='Weight decay for optimizer', type=float, default=0.01)
        
#         # Encoder/Decoder parameters

#         group.add_argument("--latent_channels", type=int, default=3, help='Output dimension for the image encoder')
        
#         # group.add_argument("--num_samples", type=int, default=4096, help='Number of samples to take from the mesh to start the encoding')
#         group.add_argument("--input_dim", type=int, default=6, help='Input dimension for the encoder')
#         group.add_argument("--output_dim", type=int, default=3, help='Output dimension of the model')
#         group.add_argument("--K", type=int, nargs="*", default=[27, 125], help='Number of K neighbors for each stage')
#         group.add_argument("--num_heads", type=int, nargs="*", default=[64, 128], help='Number of attention heads per stage the encoder')
#         group.add_argument("--stages", type=int, nargs="*", default=[64, 128], help='Dimension per stage')
#         group.add_argument("--dropout", type=float, default=0.1, help='Dropout rate')
#         group.add_argument("--pooling_factor", type=float, nargs="*", default=[0.5, 0.5], help='Pooling factor')
#         group.add_argument("--score_pooling", type=int, default=1, help='Use score base pooling')
#         group.add_argument("--pooling_hidden_dim", type=int, nargs="*", default=[32, 64], help='Hidden dim for the pooling layer')
#         group.add_argument("--feed_forward_hidden_dim", type=int, nargs="*", default=[32, 64], help='Hidden dim for the Residual FeedForward layer')
#         # group.add_argument("--use_skip_connection", type=int, default=1, help='Use skip connections, i.e., unet style network')

#         group.add_argument("--num_random_sweeps", type=int, default=3, help='How many random sweeps to use')
#         group.add_argument("--n_grids", type=int, default=200, help='Number of grids')
#         group.add_argument("--target_label", type=int, default=7, help='Target label')

#         # group.add_argument("--n_fixed_samples", type=int, default=12288, help='Number of fixed samples')

#         return parent_parser
    
#     def configure_optimizers(self):
#         optimizer = optim.AdamW(self.parameters(),
#                                 lr=self.hparams.lr,
#                                 weight_decay=self.hparams.weight_decay)
#         return optimizer

#     def on_fit_start(self):

#         # Define the file names directly without using out_dir
#         grid_t_file = 'grid_t.pt'
#         inverse_grid_t_file = 'inverse_grid_t.pt'
#         mask_fan_t_file = 'mask_fan_t.pt'

#         if not os.path.exists(grid_t_file):
#             grid_tensor = []
#             inverse_grid_t = []
#             mask_fan_t = []

#             for i in range(self.hparams.n_grids):

#                 grid_w, grid_h = self.hparams.grid_w, self.hparams.grid_h
#                 center_x = self.hparams.center_x
#                 r1 = self.hparams.r1

#                 center_y = self.hparams.center_y_start + (self.hparams.center_y_end - self.hparams.center_y_start) * (torch.rand(1))
#                 r2 = self.hparams.r2_start + ((self.hparams.r2_end - self.hparams.r2_start) * torch.rand(1)).item()
#                 theta = self.hparams.theta_start + ((self.hparams.theta_end - self.hparams.theta_start) * torch.rand(1)).item()
                
#                 grid, inverse_grid, mask = self.USR.init_grids(grid_w, grid_h, center_x, center_y, r1, r2, theta)

#                 grid_tensor.append(grid.unsqueeze(dim=0))
#                 inverse_grid_t.append(inverse_grid.unsqueeze(dim=0))
#                 mask_fan_t.append(mask.unsqueeze(dim=0))

#             self.grid_t = torch.cat(grid_tensor).to(self.device)
#             self.inverse_grid_t = torch.cat(inverse_grid_t).to(self.device)
#             self.mask_fan_t = torch.cat(mask_fan_t).to(self.device)

#             # Save tensors directly to the current directory
            
#             torch.save(self.grid_t, grid_t_file)
#             torch.save(self.inverse_grid_t, inverse_grid_t_file)
#             torch.save(self.mask_fan_t, mask_fan_t_file)

#             # print("Grids SAVED!")
#             # print(self.grid_t.shape, self.inverse_grid_t.shape, self.mask_fan_t.shape)
        
#         elif not hasattr(self, 'grid_t'):
#             # Load tensors directly from the current directory
#             self.grid_t = torch.load(grid_t_file).to(self.device)
#             self.inverse_grid_t = torch.load(inverse_grid_t_file).to(self.device)
#             self.mask_fan_t = torch.load(mask_fan_t_file).to(self.device)

#     def sampling(self, z_mu: torch.Tensor, z_sigma: torch.Tensor) -> torch.Tensor:        
#         eps = torch.randn_like(z_sigma)
#         z_vae = z_mu + eps * z_sigma
#         return z_vae
    
#     def compute_loss(self, x, Y, step="train", sync_dist=False):
        
#         loss_chamfer, _ = chamfer_distance(x, Y, batch_reduction="mean", point_reduction="sum")

#         loss = loss_chamfer

#         self.log(f"{step}_loss", loss, sync_dist=sync_dist)        
#         # self.log(f"{step}_loss_mse", loss_mse, sync_dist=sync_dist)
#         # self.log(f"{step}_loss_chamfer", loss_chamfer, sync_dist=sync_dist)
#         # self.log(f"{step}_loss_point_mesh_face", loss_point_mesh_face, sync_dist=sync_dist)
#         # self.log(f"{step}_loss_point_mesh_edge", loss_point_mesh_edge, sync_dist=sync_dist)

#         return loss
    
#     def create_mesh(self, V, F):
#         return Meshes(verts=V, faces=F)
    
#     def get_target(self, X, X_origin, X_end):
#         # put the diffusor in the fov
#         diffusor_in_fov = self.vs.diffusor_in_fov(X.float(), diffusor_origin=X_origin, diffusor_end=X_end)

#         V_fov = self.vs.fov_physical().reshape(-1, 3)
#         V_diff = []
        
#         # Get only non-background points and their corresponding labels
#         for d_fov in diffusor_in_fov:

#             d_fov = d_fov.reshape(-1)

#             V_diff.append(V_fov[d_fov == self.hparams.target_label])
        
#         # Pad them to create tensors
#         V_diff = pad_sequence(V_diff, batch_first=True, padding_value=0.0)
        
#         return V_diff

#     def volume_sampling(self, X, X_origin, X_end, use_random=False):
#         with torch.no_grad():
#             self.us_simulator_cut_td.eval()

#             probe_origin_rand = None
#             probe_direction_rand = None
#             grid = None
#             inverse_grid = None
#             mask_fan = None

#             if use_random:
#                 probe_origin_rand = torch.rand(3, device=self.device)*0.0001
#                 probe_origin_rand = probe_origin_rand
#                 rotation_ranges = ((-5, 5), (-5, 5), (-10, 10))  # ranges in degrees for x, y, and z rotations
#                 probe_direction_rand = self.vs.random_affine_matrix(rotation_ranges).to(self.device)

#                 grid_idx = torch.randint(low=0, high=self.hparams.n_grids - 1, size=(1,))
            
#                 grid = self.grid_t[grid_idx]
#                 inverse_grid = self.inverse_grid_t[grid_idx]
#                 mask_fan = self.mask_fan_t[grid_idx]

#             X_sweeps = []
#             X_sweeps_tags = []
#             for tag in np.random.choice(self.vs.tags, self.hparams.num_random_sweeps):
#                 sampled_sweep = self.vs.diffusor_sampling_tag(tag, X.to(torch.float), X_origin.to(torch.float), X_end.to(torch.float), probe_origin_rand=probe_origin_rand, probe_direction_rand=probe_direction_rand, use_random=use_random)

#                 sampled_sweep_simu = []

#                 for ss in sampled_sweep:
#                     ss_chunk = []
#                     for ss_ch in torch.chunk(ss, chunks=20, dim=1):
#                         simulated = self.us_simulator_cut_td(ss_ch.unsqueeze(dim=1), grid, inverse_grid, mask_fan)
#                         # simulated = self.us_simulator_cut_td.module.transform_us(simulated)
#                         ss_chunk.append(simulated)
#                     sampled_sweep_simu.append(torch.cat(ss_chunk, dim=2))
                
#                 sampled_sweep_simu = torch.stack(sampled_sweep_simu).detach()
#                 X_sweeps.append(sampled_sweep_simu)
#                 X_sweeps_tags.append(self.vs.tags_dict[tag])
#                 # out_fovs = self.vs.simulated_sweep_in_fov(tag, sampled_sweep_simu.detach())
#                 # X_sweeps.append(out_fovs.unsqueeze(1).unsqueeze(1).detach())
#             X_sweeps = torch.cat(X_sweeps, dim=1)
#             X_sweeps_tags = torch.tensor(X_sweeps_tags).repeat(X_sweeps.shape[0], 1)
            
#             # Y = self.us_simulator_cut_td.module.USR.mean_diffusor_dict[X.to(torch.long)].to(self.device) + torch.randn(X.shape, device=self.device) * self.us_simulator_cut_td.module.USR.variance_diffusor_dict[X.to(torch.long)].to(self.device)
#             # Y = self.vs.diffusor_in_fov(Y, X_origin, X_end)

#             return X_sweeps, X_sweeps_tags


#     def training_step(self, train_batch, batch_idx):
#         X, X_origin, X_end = train_batch

#         Y = self.get_target(X, X_origin, X_end)

#         x_sweeps, sweeps_tags = self.volume_sampling(X, X_origin, X_end, use_random=True)

#         # x_sweeps shape is B, N, C, T, H, W. N for number of sweeps ex. torch.Size([2, 2, 1, 200, 256, 256]) 
#         # tags shape torch.Size([2, 2])

#         Nsweeps = x_sweeps.shape[1]
        
#         x = []
#         x_v = []

#         for n in range(Nsweeps):
#             x_sweeps_n = x_sweeps[:, n, :, :, :, :]
#             sweeps_tags_n = sweeps_tags[:, n]

#             x_sweeps_n, x_sweeps_n_v = self.encode(x_sweeps_n, sweeps_tags_n)
            
#             x.append(x_sweeps_n)
#             x_v.append(x_sweeps_n_v)

#         x = torch.cat(x, dim=1)
#         x_v = torch.cat(x_v, dim=1)

#         x_hat = self.encode_v(x, x_v)

#         Y_v = self.get_target(X, X_origin, X_end)

#         return self.compute_loss(x_hat, Y_v, step="train")
        

#     def validation_step(self, val_batch, batch_idx):
        
#         X, X_origin, X_end = val_batch

#         x_sweeps, sweeps_tags = self.volume_sampling(X, X_origin, X_end)

#         # x_sweeps shape is B, N, C, T, H, W. N for number of sweeps ex. torch.Size([2, 2, 1, 200, 256, 256]) 
#         # tags shape torch.Size([2, 2])

#         Nsweeps = x_sweeps.shape[1]
        
#         x = []
#         x_v = []

#         for n in range(Nsweeps):
#             x_sweeps_n = x_sweeps[:, n, :, :, :, :]
#             sweeps_tags_n = sweeps_tags[:, n]

#             x_sweeps_n, x_sweeps_n_v = self.encode(x_sweeps_n, sweeps_tags_n)
            
#             x.append(x_sweeps_n)
#             x_v.append(x_sweeps_n_v)

#         x = torch.cat(x, dim=1)
#         x_v = torch.cat(x_v, dim=1)

#         x_hat = self.encode_v(x, x_v)

#         Y_v = self.get_target(X, X_origin, X_end)

#         return self.compute_loss(x_hat, Y_v, step="val", sync_dist=True)

#     def encode(self, x, sweeps_tags):

#         x = self.encoder_i(x)
        
#         x_e = self.attn_chunk(x)
        
#         x = x_e.permute(0, 1, 4, 3, 2)
        
#         x_v = self.vs.diffusor_tag_resample(x, tag=sweeps_tags)
#         x = x.flatten(2).permute(0, 2, 1)

#         x = torch.cat([x, x_v], dim=-1)

#         return x, x_v

#     def encode_v(self, x, x_v):
        
#         x_v_fixed = None

#         if hasattr(self, 'x_v_fixed'):
#             x_v_fixed = self.x_v_fixed
#             batch_size = x_v.shape[0]
#             x_v_fixed = x_v_fixed.repeat(batch_size, 1, 1)


#         x, x_v, unpooling_idx = self.encoder_v(x, x_v, x_v_fixed=x_v_fixed)

#         x = self.fc(x)

#         # z_mu = self.ff_mu(h)
#         # z_sigma = self.ff_sigma(h)
#         # z = self.sampling(z_mu, z_sigma)

#         return x

#     def forward(self, x, sweeps_tags):

#         x, x_v = self.encode(x, sweeps_tags)

#         x = self.encode_v(x, x_v)

#         # z_mu = self.ff_mu(h)
#         # z_sigma = self.ff_sigma(h)
#         # z = self.sampling(z_mu, z_sigma)

#         return x


# class USPCSegmentation(LightningModule):
#     def __init__(self, **kwargs):
#         super().__init__()
#         self.save_hyperparameters()
        
        
#         self.encoder = MHAIdxEncoder(input_dim=self.hparams.input_dim, 
#                                      output_dim=self.hparams.stages[-1], 
#                                      K=self.hparams.K, 
#                                      num_heads=self.hparams.num_heads, 
#                                      stages=self.hparams.stages, 
#                                      dropout=self.hparams.dropout, 
#                                      pooling_factor=self.hparams.pooling_factor, 
#                                      pooling_hidden_dim=self.hparams.pooling_hidden_dim,
#                                      score_pooling=self.hparams.score_pooling,
#                                      feed_forward_hidden_dim=self.hparams.feed_forward_hidden_dim, 
#                                      use_skip_connection=self.hparams.use_skip_connection)
        
#         self.decoder = MHAIdxDecoder(input_dim=self.hparams.stages[-1], 
#                                      output_dim=self.hparams.output_dim, 
#                                      K=self.hparams.K[::-1], 
#                                      num_heads=self.hparams.num_heads[::-1], 
#                                      stages=self.hparams.stages[::-1], 
#                                      dropout=self.hparams.dropout, 
#                                      pooling_hidden_dim=self.hparams.pooling_hidden_dim[::-1] if self.hparams.pooling_hidden_dim is not None else None,
#                                      feed_forward_hidden_dim=self.hparams.feed_forward_hidden_dim[::-1] if self.hparams.feed_forward_hidden_dim is not None else None,
#                                      use_skip_connection=self.hparams.use_skip_connection)
                                  
        
#         # self.loss_fn = nn.CrossEntropyLoss()
#         self.loss_fn = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
#         # self.loss_fn = nn.MSELoss()
    
#     @staticmethod
#     def add_model_specific_args(parent_parser):
#         group = parent_parser.add_argument_group("USPCSegmentation")

#         group.add_argument("--lr", type=float, default=1e-4)
#         group.add_argument('--weight_decay', help='Weight decay for optimizer', type=float, default=0.01)
        
#         # ENCODER params
#         group.add_argument("--input_dim", type=int, default=1, help='Input dimension for the encoder')
#         group.add_argument("--output_dim", type=int, default=1, help='Output dimension of the model')
#         group.add_argument("--K", type=int, nargs="*", default=[27, 27, 27, 27], help='Number of K neighbors for each stage')
#         group.add_argument("--num_heads", type=int, nargs="*", default=[2, 4, 8, 16], help='Number of attention heads per stage the encoder')
#         group.add_argument("--stages", type=int, nargs="*", default=[8, 16, 32, 64], help='Dimension per stage')
#         group.add_argument("--dropout", type=float, default=0.1, help='Dropout rate')
#         group.add_argument("--pooling_factor", type=float, nargs="*", default=[0.125, 0.5, 0.5, 0.5], help='Pooling factor')
#         group.add_argument("--score_pooling", type=int, default=1, help='Use score base pooling')
#         group.add_argument("--pooling_hidden_dim", type=int, nargs="*", default=[4, 8, 16, 32], help='Hidden dim for the pooling layer')
#         group.add_argument("--feed_forward_hidden_dim", type=int, nargs="*", default=None, help='Hidden dim for the Residual FeedForward layer')
#         group.add_argument("--use_skip_connection", type=int, default=0, help='Use skip connections, i.e., unet style network')
        
#         group.add_argument('--threshold', help='Threshold value for the simulated US, use to filter background level and reduce the number of points', type=float, default=0.0)       
#         group.add_argument('--use_v', help='Use the coordinates as well for the encoder', type=int, default=0)

#         return parent_parser
    
#     def configure_optimizers(self):
#         optimizer = optim.AdamW(self.parameters(),
#                                 lr=self.hparams.lr,
#                                 weight_decay=self.hparams.weight_decay)
#         return optimizer
    
#     def compute_loss(self, x, Y, step="train", sync_dist=False):

#         x = x.permute(0, 2, 1)
#         Y = Y.permute(0, 2, 1)

#         loss = self.loss_fn(x, Y)
        
#         self.log(f"{step}_loss", loss, sync_dist=sync_dist)

#         return loss

#     def training_step(self, train_batch, batch_idx):
#         X_d = train_batch
#         X_v, X_f = X_d["img"]
#         Y_v, Y = X_d["seg"]
        
#         x = self(X_v, X_f)

#         return self.compute_loss(x, Y, step="train")
        

#     def validation_step(self, val_batch, batch_idx):
#         X_d = val_batch
#         X_v, X_f = X_d["img"]
#         Y_v, Y = X_d["seg"]

#         x = self(X_v, X_f)

#         return self.compute_loss(x, Y, step="val", sync_dist=True)

#     def forward(self, x_v, x_f):
#         # V, VF = self.get_grid_VF(X)
#         skip_connections = None
#         if self.hparams.use_skip_connection:
#             x, unpooling_idxs, skip_connections = self.encoder(x_f, x_v)
#         else:
#             x, unpooling_idxs = self.encoder(x_f, x_v)
#         return self.decoder(x, unpooling_idxs, skip_connections)


# class USGAPC(LightningModule):
#     def __init__(self, **kwargs):
#         super().__init__()
#         self.save_hyperparameters()

#         self.encoder_i = TimeDistributed(Encoder(spatial_dims=2,
#             in_channels=1,
#             num_channels=[8, 16, 32, 64, 128],
#             out_channels=self.hparams.latent_channels,
#             num_res_blocks=[2, 2, 2, 2, 2],
#             norm_num_groups=8,
#             norm_eps=1e-3,
#             attention_levels=[False, False, False, False, True],
#             with_nonlocal_attn=True), time_dim=2)
        
#         # encoder_i = tv_models.efficientnet_v2_s().features
#         # encoder_i[0] = Conv2dNormActivation(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False, norm_layer=nn.BatchNorm2d, activation_layer=nn.SiLU)
#         # self.encoder_i = TimeDistributed(encoder_i, time_dim=2)

#         self.attn_chunk = AttentionChunk(input_dim=(self.hparams.latent_channels*16*16), hidden_dim=64, chunks=16)

#         self.encoder_v = MHAIdxEncoder(input_dim=self.hparams.input_dim, 
#                                      output_dim=self.hparams.stages[-1], 
#                                      K=self.hparams.K, 
#                                      num_heads=self.hparams.num_heads, 
#                                      stages=self.hparams.stages, 
#                                      dropout=self.hparams.dropout, 
#                                      pooling_factor=self.hparams.pooling_factor, 
#                                      pooling_hidden_dim=self.hparams.pooling_hidden_dim,
#                                      score_pooling=self.hparams.score_pooling,
#                                      feed_forward_hidden_dim=self.hparams.feed_forward_hidden_dim, 
#                                      use_skip_connection=0, 
#                                      return_v=True)

#         self.mha = Residual(nn.MultiheadAttention(embed_dim=self.hparams.stages[-1], num_heads=self.hparams.stages[-1], dropout=self.hparams.dropout))

#         if self.hparams.use_layer_norm:
#             self.norm = nn.LayerNorm(self.hparams.stages[-1])
        
#         self.attn = SelfAttention(self.hparams.stages[-1], self.hparams.pooling_hidden_dim[-1])
#         self.fc = nn.Linear(self.hparams.stages[-1], self.hparams.output_dim)
        
#         self.vs = VolumeSamplingBlindSweep(mount_point=self.hparams.mount_point)

#         self.loss_fn = nn.L1Loss()
    
#     @staticmethod
#     def add_model_specific_args(parent_parser):
#         group = parent_parser.add_argument_group("USGAPC")

#         group.add_argument("--lr", type=float, default=1e-4)
#         group.add_argument('--weight_decay', help='Weight decay for optimizer', type=float, default=0.01)
        
#         # Encoder/Decoder parameters

#         group.add_argument("--latent_channels", type=int, default=3, help='Output dimension for the image encoder')
        
#         group.add_argument("--input_dim", type=int, default=6, help='Input dimension for the encoder')
#         group.add_argument("--output_dim", type=int, default=1, help='Output dimension of the model')
#         group.add_argument("--K", type=int, nargs="*", default=[125, 125], help='Number of K neighbors for each stage')
#         group.add_argument("--num_heads", type=int, nargs="*", default=[64, 128], help='Number of attention heads per stage the encoder')
#         group.add_argument("--stages", type=int, nargs="*", default=[64, 128], help='Dimension per stage')
#         group.add_argument("--dropout", type=float, default=0.1, help='Dropout rate')
#         group.add_argument("--pooling_factor", type=float, nargs="*", default=[0.25, 0.25], help='Pooling factor')
#         group.add_argument("--score_pooling", type=int, default=1, help='Use score base pooling')
#         group.add_argument("--pooling_hidden_dim", type=int, nargs="*", default=[32, 64], help='Hidden dim for the pooling layer')
#         group.add_argument("--feed_forward_hidden_dim", type=int, nargs="*", default=[32, 64], help='Hidden dim for the Residual FeedForward layer')
#         group.add_argument("--use_layer_norm", type=int, default=1, help='Use layer norm')

#         group.add_argument("--max_sweeps", type=int, default=3, help='Max number of sweeps to use')

#         return parent_parser
    
#     def configure_optimizers(self):
#         optimizer = optim.AdamW(self.parameters(),
#                                 lr=self.hparams.lr,
#                                 weight_decay=self.hparams.weight_decay)
#         return optimizer

#     def sampling(self, z_mu: torch.Tensor, z_sigma: torch.Tensor) -> torch.Tensor:        
#         eps = torch.randn_like(z_sigma)
#         z_vae = z_mu + eps * z_sigma
#         return z_vae
    
#     def compute_loss(self, x, y, step="train", sync_dist=False):
        
#         loss = self.loss_fn(x, y)

#         self.log(f"{step}_loss", loss, sync_dist=sync_dist)

#         return loss
    
    
#     def training_step(self, train_batch, batch_idx):
#         x_d = train_batch

#         x = []
#         x_v = []

#         keys = x_d.keys()

#         for k in keys:
#             if isinstance(k, int):
#                 x_img = x_d[k]
#                 tags = x_d["tag"][:,k]

#                 x_, x_v_, _ = self.encode(x_img, tags)

#                 x.append(x_)
#                 x_v.append(x_v_)

#         x = torch.cat(x, dim=1)
#         x_v = torch.cat(x_v, dim=1)
        
#         x, x_s = self.attn(x, x)
#         x = self.fc(x)

#         y = x_d["ga_boe"]

#         return self.compute_loss(x, y, step="train")
        

#     def validation_step(self, val_batch, batch_idx):
        
#         x_d = val_batch
        
#         x = []
#         x_v = []

#         keys = x_d.keys()

#         for k in keys:
#             if isinstance(k, int):
#                 x_img = x_d[k]
#                 tags = x_d["tag"][:,k]

#                 x_, x_v_, _ = self.encode(x_img, tags)

#                 x.append(x_)
#                 x_v.append(x_v_)

#         x = torch.cat(x, dim=1)
#         x_v = torch.cat(x_v, dim=1)
        
#         x, x_s = self.attn(x, x)
#         x = self.fc(x)

#         y = x_d["ga_boe"]

#         return self.compute_loss(x, y, step="val", sync_dist=True)
    
#     def encode(self, x, sweeps_tags):

        
#         x = self.encoder_i(x)
        
#         x_e = self.attn_chunk(x)
        
#         x = x_e.permute(0, 1, 4, 3, 2)
        
#         x_v = self.vs.diffusor_tag_resample(x, tag=sweeps_tags)
#         x = x.flatten(2).permute(0, 2, 1)

#         x = torch.cat([x, x_v], dim=-1)

#         x, x_v, unpooling_idx = self.encoder_v(x, x_v)

#         x, x_w = self.mha(x, x, x)
        
#         return x, x_v, x_e
    
#     def forward(self, x, sweeps_tags):
        
#         x, x_v, _ = self.encode(x, sweeps_tags)

#         x, x_s = self.attn(x, x)
#         x = self.fc(x)

#         return x, x_v