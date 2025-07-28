import SimpleITK as sitk
import os


parent_dir = "/mnt/raid/C1_ML_Analysis/simulated_data_export/placenta_simu/IB1_label11"
# for folder in os.listdir(parent_dir):
#     folder_path = os.path.join(parent_dir,folder)

for file in os.listdir(parent_dir):
    if "C4_bmap_guided.nrrd" in file :
        file_path = os.path.join(parent_dir,file)
        new_folder_path = file_path.replace(".nrrd","") 


        os.makedirs(new_folder_path, exist_ok=True)

        img = sitk.ReadImage(file_path)
        video = sitk.GetArrayFromImage(img)

        for idx, frame in enumerate(video):

            output = sitk.GetImageFromArray(frame)
            output.SetSpacing(img.GetSpacing()[0:2])
            output.SetOrigin(img.GetOrigin()[0:2])

            writer = sitk.ImageFileWriter()
            writer.SetFileName(os.path.join(new_folder_path, str(idx) + ".nrrd"))
            writer.UseCompressionOn()
            writer.Execute(output)
