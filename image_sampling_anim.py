import os

import pandas as pd
import numpy as np
import SimpleITK as sitk

import torch
import glob 

import dl.nets.us_simulation_jit as us_simulation_jit
import dl.nets.us_simu as us_simu

from dl.nets.layers import TimeDistributed

import argparse


def main(args):

    if not os.path.exists(args.out):
        os.makedirs(args.out)
    
    mount_point = args.mount_point

    vs = us_simu.VolumeSamplingBlindSweep(mount_point=mount_point, simulation_fov_fn='simulation_fov.stl', simulation_ultrasound_plane_fn='ultrasound_grid.stl')
    vs.init_probe_params_from_pos(args.probe_paths)
    vs = vs.cuda()

    diffusor = sitk.ReadImage(args.img)
    diffusor_t = torch.tensor(sitk.GetArrayFromImage(diffusor).astype(int))
    diffusor_size = torch.tensor(diffusor.GetSize())
    diffusor_spacing = torch.tensor(diffusor.GetSpacing())
    diffusor_origin = torch.tensor(diffusor.GetOrigin())
    diffusor_end = diffusor_origin + diffusor_spacing * diffusor_size
    
    diffusor_batch_t = diffusor_t.cuda().float().unsqueeze(0).unsqueeze(0)

    diffusor_origin_batch = diffusor_origin[None, :].cuda()
    diffusor_end_batch = diffusor_end[None, :].cuda()

    
    simulator = us_simulation_jit.MergedGuidedAnim()
    simulator = simulator.cuda()
    grid, inverse_grid, mask_fan = simulator.init_grids(256, 256, 128.0, -30.0, 20.0, 215.0, 0.7853981633974483) # w, h, center_x, center_y, r1, r2, theta
    simulator = TimeDistributed(simulator, time_dim=2).eval().cuda()

    for tag in vs.tags:

        sampled_sweep_simu_t = vs.get_sweep(diffusor_batch_t.to(torch.float).cuda(), diffusor_origin_batch.to(torch.float).cuda(), diffusor_end_batch.to(torch.float).cuda(), tag, use_random=False, simulator=simulator)
        sampled_sweep_simu_np = sampled_sweep_simu_t.squeeze().cpu().numpy()

        #Save image
        out_path = os.path.join(args.out, f'{tag}.nrrd')
        
        print(f"Writing sampled sweep {tag} to {out_path}")
        sampled_sweep_simu_img = sitk.GetImageFromArray(sampled_sweep_simu_np)
        sitk.WriteImage(sampled_sweep_simu_img, out_path)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--img', type=str, required=True, help='Path to the labeled diffusor image')
    parser.add_argument('--probe_paths', type=str, required=True, help='Path to the probe paths file')
    parser.add_argument('--out', type=str, required=True, help='Path to the output directory to save the sweeps')
    parser.add_argument('--mount_point', type=str, default='/mnt/raid/C1_ML_Analysis/simulated_data_export/animation_export/', help='Mount point for the simulation assets')
    parser.add_argument('--ow', type=int, default=0, help='Overwrite existing files: 0 for no, 1 for yes')
    
    args = parser.parse_args()
    if args.ow == 0 and os.path.exists(args.out):
        print(f"Output directory {args.out} already exists. Use --ow 1 to overwrite.")
        exit(1)
    main(args)






