import torch
import multiprocessing as mp
from image_generation import image_generation

# rsync -a

def multi_gpu_image_gen(gpu_id, vid_path):
    torch.cuda.set_device(gpu_id)

    print(f"GPU {gpu_id}: Processing {vid_path}")
    image_generation(vid_path, 20, 30000, "/mnt/raid/home/ajarry/data/cephalic_output")

if __name__ == "__main__":
    # List of CSVs, each for a different GPU
    vid_paths = [
        '/mnt/raid/home/ajarry/data/all_poses_sweeps_us/frame_0206/L0.nrrd',
        '/mnt/raid/home/ajarry/data/all_poses_sweeps_us/frame_0206/L1.nrrd',
        '/mnt/raid/home/ajarry/data/all_poses_sweeps_us/frame_0206/R0.nrrd',
        '/mnt/raid/home/ajarry/data/all_poses_sweeps_us/frame_0206/R1.nrrd',
    ]

    processes = []
    for gpu_id in range(4):
        p = mp.Process(target=multi_gpu_image_gen, args=(gpu_id, vid_paths[gpu_id]))
        p.start()
        processes.append(p)

    # Wait for all to complete
    for p in processes:
        p.join()
