import os
import csv

origin = '/mnt/raid/home/ajarry/data/all_poses_sweeps_us'
l =[]
for folder in sorted(os.listdir(origin)[1101:1200]):
    path_folder = os.path.join(origin,folder)
    for images in sorted(os.listdir(path_folder)):
        image = os.path.join(path_folder,images)
        l.append(image)

print(l)
with open("1101to1200.csv", "w", newline="") as f:
    writer = csv.writer(f)
    for line in l:
        writer.writerow([line])  # wrap in list to make it a single column

check_path = "/mnt/raid/home/ajarry/data/cephalic_output"
folder_list = sorted(os.listdir(check_path))
for i in range(len(folder_list)):
    if int(folder_list[i].split("_")[-1]) +1 != int(folder_list[i+1].split("_")[-1]):
        print(folder_list[i])