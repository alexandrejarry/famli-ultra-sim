import os
import sys
import SimpleITK as sitk
import numpy as np
import nrrd
import torch
sys.path.append("/mnt/raid/home/ajarry/data/famli-ultra-sim")
sys.path.append("/mnt/raid/home/ajarry/data/famli-ultra-sim/dl")
sys.path.append("/mnt/raid/home/ajarry/data/famli-ultra-sim/dl/nets")

from dl.nets.layers import TimeDistributed
import dl.nets.us_simulation_jit as us_simulation_jit
from dl.nets.us_simu import VolumeSamplingBlindSweep


folder_path = "/mnt/raid/C1_ML_Analysis/simulated_data_export/placenta/"
folder_save_path = "/mnt/raid/C1_ML_Analysis/simulated_data_export/placenta_simu/"
keyword = "label11.nrrd"
def get_files_with_name(folder_path, keyword):
    matching_files = []
    for filename in os.listdir(folder_path):
        if keyword in filename:
            matching_files.append(filename)
    return matching_files

folder_list = get_files_with_name(folder_path,keyword)

mount_point = "/mnt/raid/C1_ML_Analysis/"
i = 0
for file in folder_list:
    i+=1
    print("Folder number: ", i)
    extracted_name = file.replace("_label11.nrrd","")
    folders_path = os.path.join(folder_save_path,extracted_name)

    diffusor_np, diffusor_head = nrrd.read(os.path.join(mount_point, "simulated_data_export/placenta/" + file))

    diffusor_size = diffusor_head['sizes']
    diffusor_spacing = np.diag(diffusor_head['space directions'])

    diffusor_origin = np.flip(diffusor_head['space origin'], axis=0)
    diffusor_end = diffusor_origin + diffusor_spacing * diffusor_size
    diffusor_t = torch.tensor(diffusor_np.astype(float)).unsqueeze(0).unsqueeze(0).cuda()
    diffusor_origin = torch.tensor(diffusor_origin.copy()).unsqueeze(0).cuda()
    diffusor_end = torch.tensor(diffusor_end.copy()).unsqueeze(0).cuda()

    print("Opened file: ", file)

    # us_simulator = us_simulation_jit.MergedLinearLabel11().eval().cuda()
    # us_simulator = us_simulation_jit.MergedLinearLabel11WOG().eval().cuda()
    us_simulator = us_simulation_jit.MergedLinearLabel11PassThrough().eval().cuda()



    grid, inverse_grid, mask_fan = us_simulator.init_grids(256, 256, 128.0, -30.0, 20.0, 215.0, 0.7853981633974483)
    grid = grid.cuda()
    inverse_grid = inverse_grid.cuda()
    mask_fan = mask_fan.cuda()

    us_simulator_td = TimeDistributed(us_simulator, time_dim=2).eval()

    vs = VolumeSamplingBlindSweep(mount_point=mount_point).cuda()

    for tag_idx in range(len(vs.tags)):
        sweep = vs.get_sweep(diffusor_t, diffusor_origin, diffusor_end, vs.tags[tag_idx], use_random=False, simulator=us_simulator_td, grid=grid, inverse_grid=inverse_grid, mask_fan=mask_fan)
        img = sitk.GetImageFromArray(sweep.squeeze().detach().cpu().numpy())
        img.SetSpacing([0.75, 0.75, 1.0])
        print("Saved : ", folders_path + "_label11/" + vs.tags[tag_idx] + "_label.nrrd")
        sitk.WriteImage(img, folders_path + "_label11/" + vs.tags[tag_idx] + "_label.nrrd")

