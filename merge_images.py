import os
import numpy as np
import SimpleITK as sitk
import argparse
import pandas as pd

from multiprocessing import Pool, cpu_count

class MergeGroup():
    def __init__(self, df_g, output_size, output_spacing, output_origin, out_dir, mount_dir = "./"):
        self.df_g = df_g
        self.output_size = output_size
        self.output_spacing = output_spacing
        self.output_origin = output_origin
        self.out_dir = out_dir
        self.mount_dir = mount_dir

    def __call__(self, g_name):

        out_fn = os.path.join(self.out_dir, g_name + ".nrrd")                
        out_diffusor_fn = os.path.join(self.out_dir, g_name + "_diffusor.nrrd")                

        if not os.path.exists(out_fn):

            print("Group merge:", g_name)
            g = self.df_g.get_group(g_name)

            out_np = np.zeros(self.output_size, dtype=np.ushort)
            out_img = sitk.GetImageFromArray(out_np)
            out_img.SetSpacing(self.output_spacing)
            out_img.SetOrigin(self.output_origin)

            out_diffusor_np = np.zeros(self.output_size, dtype=np.ushort)            

            for idx, row in g.iterrows():
            
                # print("Processing:", row)

                label = row['label']
                input_img_filename = os.path.join(self.mount_dir, row['img'])
                    
                img = sitk.ReadImage(input_img_filename)

                InterpolatorType = sitk.sitkNearestNeighbor
                resampleImageFilter = sitk.ResampleImageFilter()
                resampleImageFilter.SetInterpolator(InterpolatorType)
                resampleImageFilter.SetReferenceImage(out_img)
                resampled_img = resampleImageFilter.Execute(img)

                resampled_img_np = sitk.GetArrayFromImage(resampled_img).astype(np.ushort)

                out_np[resampled_img_np > 0] = (resampled_img_np*(label))[resampled_img_np > 0]

                mean = np.abs(row[args.mean])
                std = np.abs(row[args.std])

                out_diffusor_np[resampled_img_np > 0] = (np.clip(np.random.normal(loc=mean, scale=std, size=out_np.shape), a_min=args.a_min, a_max=args.a_max))[resampled_img_np > 0]          

            out_img = sitk.GetImageFromArray(out_np)
            out_img.SetSpacing(self.output_spacing)
            out_img.SetOrigin(self.output_origin)

            sitk.WriteImage(out_img, out_fn)

            
            out_diffusor_img = sitk.GetImageFromArray(out_diffusor_np)
            out_diffusor_img.SetSpacing(self.output_spacing)
            out_diffusor_img.SetOrigin(self.output_origin)

            sitk.WriteImage(out_diffusor_img, out_diffusor_fn)


def merge_groups(args, output_size, output_spacing, output_origin, out_dir):    

    output_img_diffusor_fn = os.path.normpath(out_dir) + "_diffusor.nrrd"

    if args.ow or not os.path.exists(output_img_diffusor_fn):
    
        output_img_fn = os.path.normpath(out_dir) + ".nrrd"
        out_np = np.zeros(output_size, dtype=np.ushort)
        out_diffusor_np = np.random.normal(loc=args.b_mean, scale=args.b_std, size=output_size)

        print("Merging groups...")

        for g_name in args.group_order:

            print(g_name)
            g_image_fn = os.path.join(out_dir, g_name + ".nrrd")
            g_img_np = sitk.GetArrayFromImage(sitk.ReadImage(g_image_fn))

            out_np[g_img_np > 0] = g_img_np[g_img_np > 0]

            g_image_diffusor_fn = os.path.join(out_dir, g_name + "_diffusor.nrrd")
            g_img_diffusor_np = sitk.GetArrayFromImage(sitk.ReadImage(g_image_diffusor_fn))

            out_diffusor_np[g_img_np > 0] = g_img_diffusor_np[g_img_np > 0]    

        out_img = sitk.GetImageFromArray(out_np)
        out_img.SetSpacing(output_spacing)
        out_img.SetOrigin(output_origin)

        print(out_img)
        
        print("Writing:", output_img_fn)
        writer = sitk.ImageFileWriter()
        writer.SetFileName(output_img_fn)
        writer.UseCompressionOn()
        writer.Execute(out_img)

        out_diffusor_np = np.clip(out_diffusor_np, a_min=args.a_min, a_max=args.a_max)
        out_diffusor = sitk.GetImageFromArray(out_diffusor_np)
        out_diffusor.SetSpacing(output_spacing)
        out_diffusor.SetOrigin(output_origin)
        
        print("Writing:", output_img_diffusor_fn)
        writer = sitk.ImageFileWriter()
        writer.SetFileName(output_img_diffusor_fn)
        writer.UseCompressionOn()
        writer.Execute(out_diffusor)

def main(args):
    
    min_size = args.min_size

    origin_np = []
    spacing_np = []
    size_np = []

    df = pd.read_csv(args.csv)

    for idx, row in df.iterrows():
        input_img_filename = os.path.join(args.mount_dir, str(row['img']))

        if os.path.exists(input_img_filename):

            ignore_size_info = False
            if "ignore_size_info" in row and row["ignore_size_info"]:
                ignore_size_info = True
            
            if not ignore_size_info:
                print("Processing:", input_img_filename)
                reader = sitk.ImageFileReader()
                reader.SetFileName(input_img_filename)
                reader.ReadImageInformation()
                origin = reader.GetOrigin()
                spacing = reader.GetSpacing()
                size = reader.GetSize()
                print(origin, spacing, size)

                origin_np.append(origin)
                spacing_np.append(spacing)
                size_np.append(size)

    origin_np = np.array(origin_np)
    spacing_np = np.array(spacing_np)
    size_np = np.array(size_np)

    output_origin = np.min(origin_np, axis=0)
    output_end = np.max(origin_np + np.multiply(size_np, spacing_np), axis=0)

    if args.pad > 0:
        pad = args.pad*(output_end - output_origin)
        output_origin -= pad
        output_end += pad

    print(output_origin, output_end)
    min_spc = np.min(np.abs((output_end - output_origin))/min_size)

    output_spacing = np.array([min_spc, min_spc, min_spc])
    output_size = np.floor(np.abs((output_end - output_origin))/output_spacing).astype(int)
    print("Output image info:")
    print("size", output_size)
    print("spacing", output_spacing)
    print("origin", output_origin)

    output_size = np.flip(output_size)
    # output_spacing = np.flip(output_spacing)
    # output_origin = np.flip(output_origin)
    
    df_g = df.groupby('group')

    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = args.mount_dir
    print("Merged images output directory:", out_dir)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    cpu_count = args.n_proc if args.n_proc is not None else cpu_count()

    with Pool(cpu_count) as p:
        p.map(MergeGroup(df_g, output_size, output_spacing, output_origin, out_dir, args.mount_dir), args.group_order)

    merge_groups(args, output_size, output_spacing, output_origin, out_dir)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Merge NRRD files as a single image', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--csv', type=str, help='CSV file with columns img,group - column ignore_size_info to ignore the size during compute of image space is optional', required=True)
    parser.add_argument('--out_dir', type=str, help='Output directory', default=None)
    parser.add_argument('--mount_dir', type=str, help='Mount directory/input directory', default="./")
    parser.add_argument('--group_order', type=str, nargs='+', help='Group order to merge. Latter values replace preceding values.', default=
        # ["lady",
        #"uterus",
        # ["gestational",
        ["lady",
        "lady_sacrum",
        "lady_bladder",
        "lady_uterus",
        "lady_sac",
        "fetus",
        "visceral",
        "bronchus",
        "brain",        
        "cardiovascular",
        "skull",
        "skeleton",
        "ribs",
        "arms",
        "legs"])
    parser.add_argument('--min_size', type=int, help='Output size', default=384)
    parser.add_argument('--pad', type=float, help='Pad the output', default=0)
    parser.add_argument('--mean', type=str, help='Name of column to replace the labeled image, it represents the mean value', default='mean')
    parser.add_argument('--std', type=str, help='Name of column to replace the labeled image, it represents the standard deviation value', default='stddev')
    parser.add_argument('--b_mean', type=float, help='Background mean', default=3.0)
    parser.add_argument('--b_std', type=float, help='Background std', default=3.0)
    parser.add_argument('--a_min', type=float, help='Clip minimum value', default=None)
    parser.add_argument('--a_max', type=float, help='Clip maximum value', default=None)
    parser.add_argument('--n_proc', type=int, help='Max number of processes', default=None)
    parser.add_argument('--ow', type=int, help='Overwrite', default=0)

    
    args = parser.parse_args()

    main(args)
