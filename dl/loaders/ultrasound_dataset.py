from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import SimpleITK as sitk
from PIL import Image
import nrrd
import os
import sys
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


from lightning.pytorch.core import LightningDataModule

import monai
from monai.transforms import (    
    LoadImage,
    LoadImaged
)
from monai.data import ITKReader

from transforms import ultrasound_transforms

import pickle

import vtk
import random


from torch.nn import functional as F

sys.path.append('/mnt/raid/C1_ML_Analysis/source/ShapeAXI/src')

from shapeaxi import utils
from shapeaxi import saxi_transforms

class USDataset(Dataset):
    def __init__(self, df, mount_point = "./", transform=None, img_column="img_path", class_column=None, ga_column=None, scalar_column=None, repeat_channel=True, return_head=False):
        self.df = df
        self.mount_point = mount_point
        self.transform = transform
        self.img_column = img_column
        self.class_column = class_column
        self.ga_column = ga_column
        self.scalar_column = scalar_column
        self.repeat_channel = repeat_channel
        self.return_head = return_head

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.mount_point, self.df.iloc[idx][self.img_column])
        
        try:
            if os.path.splitext(img_path)[1] == ".nrrd":
                img, head = nrrd.read(img_path, index_order="C")
                img = img.astype(float)
                # img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
                img = torch.tensor(img, dtype=torch.float32)
                img = img.squeeze()
                if self.repeat_channel:
                    img = img.unsqueeze(0).repeat(3,1,1)
            else:
                img = np.array(Image.open(img_path))
                img = torch.tensor(img, dtype=torch.float32)
                if len(img.shape) == 3:                    
                    img = torch.permute(img, [2, 0, 1])[0:3, :, :]
                else:                    
                    img = img.unsqueeze(0).repeat(3,1,1)            
        except:
            print("Error reading frame:" + img_path, file=sys.stderr)
            img = torch.tensor(np.zeros([3, 256, 256]), dtype=torch.float32)

        if(self.transform):
            img = self.transform(img)

        if self.class_column:
            return img, torch.tensor(self.df.iloc[idx][self.class_column]).to(torch.long)

        if self.ga_column:
            ga = self.df.iloc[idx][self.ga_column]
            return img, torch.tensor([ga])

        if self.scalar_column:
            scalar = self.df.iloc[idx][self.scalar_column]
            return img, torch.tensor(scalar)            
        if self.return_head:
            return img, head

        return img

class USDatasetV2(Dataset):
    def __init__(self, df, mount_point = "./", transform=None, img_column="img_path", class_column=None, ga_column=None, scalar_column=None, return_head=False):
        self.df = df
        self.mount_point = mount_point
        self.transform = transform
        self.img_column = img_column
        self.class_column = class_column
        self.ga_column = ga_column
        self.scalar_column = scalar_column
        self.return_head = return_head
        self.loader = LoadImage()

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.mount_point, self.df.iloc[idx][self.img_column])
        
        img = self.loader(img_path)        

        if(self.transform):
            img = self.transform(img)

        if self.class_column:
            return img, torch.tensor(self.df.iloc[idx][self.class_column]).to(torch.long)

        if self.ga_column:
            ga = self.df.iloc[idx][self.ga_column]
            return img, torch.tensor([ga])

        if self.scalar_column:
            scalar = self.df.iloc[idx][self.scalar_column]
            return img, torch.tensor(scalar)            
        if self.return_head:
            return img, head

        return img

class SimuDataset(Dataset):
    def __init__(self, df, mount_point = "./", transform=None, img_column="img_path", class_column=None, ga_column=None, scalar_column=None, repeat_channel=True, return_head=False, target_column=None, target_transform=None):
        self.df = df
        self.mount_point = mount_point
        self.transform = transform
        self.img_column = img_column
        self.target_column = target_column
        self.target_transform = target_transform
        self.class_column = class_column
        self.ga_column = ga_column
        self.scalar_column = scalar_column
        self.repeat_channel = repeat_channel
        self.return_head = return_head

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.mount_point, self.df.iloc[idx][self.img_column])
        
        try:
            if os.path.splitext(img_path)[1] == ".nrrd":
                img, head = nrrd.read(img_path, index_order="C")
                img = img.astype(float)
                # img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
                img = torch.tensor(img, dtype=torch.float32)
                img = img.squeeze()
                if self.repeat_channel:
                    img = img.unsqueeze(0).repeat(3,1,1)
            else:
                img = np.array(Image.open(img_path))
                img = torch.tensor(img, dtype=torch.float32)
                if len(img.shape) == 3:                    
                    img = torch.permute(img, [2, 0, 1])[0:3, :, :]
                else:                    
                    img = img.unsqueeze(0).repeat(3,1,1)            
        except:
            print("Error reading frame:" + img_path, file=sys.stderr)
            img = torch.tensor(np.zeros([1, 256, 256]), dtype=torch.float32)

        img = img/339
        if(self.transform):
            img = self.transform(img)

        if self.class_column:
            return img, torch.tensor(self.df.iloc[idx][self.class_column]).to(torch.long)

        if self.ga_column:
            ga = self.df.iloc[idx][self.ga_column]
            return img, torch.tensor([ga])

        if self.scalar_column:
            scalar = self.df.iloc[idx][self.scalar_column]
            return img, torch.tensor(scalar)            
        if self.return_head:
            return img, head

        if self.target_column:
            try:
                target_img_path = os.path.join(self.mount_point, self.df.iloc[idx][self.target_column])
                if os.path.splitext(img_path)[1] == ".nrrd":
                    target, head = nrrd.read(target_img_path, index_order="C")
                    # img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
                    target = torch.tensor(target, dtype=torch.float32)
                    target = target.squeeze()
                    target = target[:,:,0]
                    if self.repeat_channel:
                        target = target.unsqueeze(0).repeat(3,1,1)
                else:
                    target = np.array(Image.open(target_img_path))
                    target = torch.tensor(target, dtype=torch.float32)
                    if len(img.shape) == 3:                    
                        target = torch.permute(img, [2, 0, 1])[0:3, :, :]
                    else:                    
                        target = target.unsqueeze(0).repeat(3,1,1)            
            except:
                print("Error reading frame: " + target_img_path, file=sys.stderr)
                target = torch.tensor(np.zeros([1, 256, 256]), dtype=torch.float32)
            
            target = target/255
            if(self.transform):
                target = self.transform(target)

            return img, target

        return img


class LotusDataset(Dataset):
    def __init__(self, df, mount_point = "./", img_column="img_path", seg_column="seg_path"):
        self.df = df
        self.mount_point = mount_point
        self.img_column = img_column
        self.seg_column = seg_column
        
        self.loader = LoadImaged(keys=["img", "seg"])

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.mount_point, self.df.iloc[idx][self.img_column])
        seg_path = os.path.join(self.mount_point, self.df.iloc[idx][self.seg_column])

        d = {"img": img_path, "seg": seg_path}
        
        return self.loader(d)
    
class LotusDataModule(LightningDataModule):
    def __init__(self, df_train, df_val, df_test, mount_point="./", batch_size=256, num_workers=4, img_column="img_path", seg_column="seg_path", train_transform=None, valid_transform=None, test_transform=None, drop_last=False):
        super().__init__()

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.mount_point = mount_point
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_column = img_column        
        self.seg_column = seg_column        
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform
        self.drop_last=drop_last        

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = monai.data.Dataset(data=LotusDataset(self.df_train, self.mount_point, img_column=self.img_column, seg_column=self.seg_column), transform=self.train_transform)
        self.val_ds = monai.data.Dataset(data=LotusDataset(self.df_val, self.mount_point, img_column=self.img_column, seg_column=self.seg_column), transform=self.valid_transform)
        self.test_ds = monai.data.Dataset(data=LotusDataset(self.df_test, self.mount_point, img_column=self.img_column, seg_column=self.seg_column), transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, shuffle=True, prefetch_factor=2)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=False, pin_memory=True, drop_last=self.drop_last)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last)

class USDatasetBlindSweep(Dataset):
    def __init__(self, df, mount_point = "./", num_frames=0, img_column='img_path', ga_column=None, transform=None, id_column=None, max_sweeps=4):
        self.df = df
        self.mount_point = mount_point
        self.num_frames = num_frames
        self.transform = transform
        self.img_column = img_column
        self.ga_column = ga_column
        self.id_column = id_column
        self.max_sweeps = max_sweeps

        self.keys = self.df.index

        if self.id_column:        
            self.df_group = self.df.groupby(id_column)            
            self.keys = list(self.df_group.groups.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):

        if self.id_column:
            df_group = self.df_group.get_group(self.keys[idx])
            ga = float(df_group[self.ga_column].unique()[0])
        
            img = self.create_seq(df_group)
        
            return img, torch.tensor([ga], dtype=torch.float32)
        else:
        
            img_path = os.path.join(self.mount_point, self.df.iloc[idx][self.img_column])

            try:
                # img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
                img, head = nrrd.read(img_path, index_order="C")
                img = torch.tensor(img, dtype=torch.float32)
                if self.num_frames > 0:
                    idx = torch.randint(low=0, high=img.shape[0], size=self.num_frames)
                    # idx = torch.randperm(img.shape[0])[:self.num_frames]
                    if self.num_frames == 1:
                        img = img[idx[0]]
                    else:
                        img = img[idx]
            except:
                print("Error reading cine: " + img_path)
                if self.num_frames == 1:
                    img = torch.zeros(256, 256, dtype=torch.float32)
                else:
                    img = torch.zeros(self.num_frames, 256, 256, dtype=torch.float32)
            
            if self.num_frames == 1:
                img = img.unsqueeze(0).repeat(3,1,1).contiguous()
            else:
                img = img.unsqueeze(1).repeat(1,3,1,1).contiguous()

            if self.transform:
                img = self.transform(img)

            if self.ga_column:
                ga = self.df.iloc[idx][self.ga_column]
                return img, torch.tensor([ga])

            return img

    def create_seq(self, df):

        # shuffle
        df = df.sample(frac=1)

        # get maximum number of samples, -1 uses all
        max_sweeps = len(df.index)
        if self.max_sweeps > -1:
            max_sweeps = min(max_sweeps, self.max_sweeps)        

        # get the rows from the shuffled dataframe and sort them
        df = df[0:max_sweeps].sort_index()

        # read all of them
        
        imgs = []

        for idx, row in df.iterrows():
            # try:
            img_path = os.path.join(self.mount_point, row[self.img_column])                
            img_np, head = nrrd.read(img_path, index_order="C")
            img_t = torch.tensor(img_np)

            if self.transform:
                img_t = self.transform(img_t)                
            imgs.append(img_t)
            # except Exception as e:
            #     print(e, file=sys.stderr)
        return torch.cat(imgs)
    
class USDatasetBlindSweepWTag(Dataset):
    def __init__(self, df, mount_point = "./", img_column='file_path', tag_column='tag', ga_column='ga_boe', transform=None, id_column='study_id', max_sweeps=3):
        self.df = df
        self.mount_point = mount_point
        self.transform = transform
        self.img_column = img_column
        self.ga_column = ga_column
        self.id_column = id_column
        self.max_sweeps = max_sweeps
        self.tag_column = tag_column

        self.keys = self.df.index

        if self.id_column:        
            self.df_group = self.df.groupby(id_column)            
            self.keys = list(self.df_group.groups.keys())

        self.tags_dict = {'M': 0,
            'L0': 1,
            'L1': 2,
            'R0': 3,
            'R1': 4,
            'C1': 5,
            'C2': 6,
            'C3': 7,
            'C4': 8}

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):

        
        df_group = self.df_group.get_group(self.keys[idx])
        ga = float(df_group[self.ga_column].unique()[0])
    
        img_d = self.create_seq(df_group)
        img_d['ga_boe'] = torch.tensor([ga], dtype=torch.float32)
    
        return img_d

    def create_seq(self, df):

        # shuffle
        df = df.sample(frac=1)

        # get maximum number of samples, -1 uses all
        max_sweeps = len(df.index)
        if self.max_sweeps > -1:
            max_sweeps = min(max_sweeps, self.max_sweeps)        

        # get the rows from the shuffled dataframe and sort them
        df = df[0:max_sweeps].sort_index()

        # read all of them
        
        imgs = {}

        imgs['tag'] = []

        num_sweeps = 0

        for idx, row in df.iterrows():
            
            img_path = os.path.join(self.mount_point, row[self.img_column])                
            img_np, head = nrrd.read(img_path, index_order="C")

            if len(img_np.shape) == 4:
                img_np = img_np[:,:,:,0]

            img_t = torch.tensor(img_np)

            if self.transform:
                img_t = self.transform(img_t)

            imgs[num_sweeps] = img_t
            num_sweeps += 1

            imgs['tag'].append(self.tags_dict[row[self.tag_column]])

        for k in range(self.max_sweeps):
            if k not in imgs:
                imgs[k] = torch.zeros(16, 256, 256, dtype=torch.float32)
                imgs['tag'].append(0)
        
        imgs['tag'] = torch.tensor(imgs['tag'], dtype=torch.long)
        
        return imgs

class USDatasetVolumes(Dataset):
    def __init__(self, df, mount_point = "./", num_frames=0, img_column='img_path', ga_column='ga_boe', id_column='study_id', max_seq=-1, transform=None):
        self.df = df
        
        self.mount_point = mount_point
        self.num_frames = num_frames
        self.transform = transform
        self.img_column = img_column
        self.ga_column = ga_column
        self.id_column = id_column
        self.max_seq = max_seq

        print(id_column)
        self.df_group = self.df.groupby(id_column)
        print(len(self.df_group.groups.keys()))
        self.keys = list(self.df_group.groups.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        
        df_group = self.df_group.get_group(self.keys[idx])
        ga = float(df_group[self.ga_column].unique()[0])
        
        img = self.create_seq(df_group)
        
        return torch.tensor(img, dtype=torch.float32), torch.tensor([ga], dtype=torch.float32)

    def create_seq(self, df):

        # shuffle
        df = df.sample(frac=1)

        # get maximum number of samples, -1 uses all
        max_seq = len(df.index)
        if self.max_seq > -1:
            max_seq = min(max_seq, self.max_seq)        

        # get the rows from the shuffled dataframe and sort them
        df = df[0:max_seq].sort_index()

        # read all of them
        imgs = []
        time_steps = 0 
        for idx, row in df.iterrows():
            try:
                img_path = os.path.join(self.mount_point, row[self.img_column])
                img_np, head = nrrd.read(img_path, index_order="C")

                if self.transform:
                    img_np = self.transform(img_np)

                imgs.append(img_np)
            except Exception as e:
                print(e, file=sys.stderr)

        return np.stack(imgs)

    # def has_all_types(self, df, seqo):
    #     if(seqo[0] == "all"):
    #         return True
    #     seq_found = np.zeros(len(seqo))
    #     for i, t in df["tag"].items():
    #         scan_index = np.where(np.array(seqo) == t)[0]
    #         for s_i in scan_index:
    #             seq_found[s_i] += 1
    #     return np.prod(seq_found) > 0

# class ITKImageDataset(Dataset):
#     def __init__(self, csv_file, transform=None, target_transform=None):
#         self.df = pd.read_csv(csv_file)

#         self.transform = transform
#         self.target_transform = target_transform
#         self.sequence_order = ["all"]

#         self.df = self.df.groupby('study_id').filter(lambda x: has_all_types(x, self.sequence_order))

#         self.df_group = self.df.groupby('study_id')
#         self.keys = list(self.df_group.groups.keys())
#         self.data_frames = []

#     def __len__(self):
#         return len(self.df_group)

#     def __getitem__(self, idx):

#         df_group = self.df_group.get_group(self.keys[idx])

#         seq_np, df = create_seq(df_group, self.sequence_order)
#         ga = df["ga_boe"]
#         # img = self.df.iloc[idx]['file_path']
#         # ga = self.df.iloc[idx]['ga_boe']

#         # reader = ITKReader()
#         # img = reader.read(img)

#         # if self.transform:
#         #     img = self.transform(img)
#         # if self.target_transform:
#         #     ga = self.target_transform(ga)

#         self.data_frames.append(df)

#         return (self.transform(seq_np), np.array([ga]))


class USDataModule(LightningDataModule):
    def __init__(self, df_train, df_val, df_test, mount_point="./", batch_size=256, num_workers=4, img_column="img_path", class_column=None, ga_column=None, scalar_column=None, train_transform=None, valid_transform=None, test_transform=None, drop_last=False, repeat_channel=True, target_column=None):
        super().__init__()

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.mount_point = mount_point
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_column = img_column
        self.class_column = class_column
        self.scalar_column = scalar_column
        self.ga_column = ga_column
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform
        self.drop_last=drop_last
        self.repeat_channel = repeat_channel
        self.target_column = target_column

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = USDataset(self.df_train, self.mount_point, img_column=self.img_column, class_column=self.class_column, ga_column=self.ga_column, scalar_column=self.scalar_column, transform=self.train_transform, repeat_channel=self.repeat_channel)
        self.val_ds = USDataset(self.df_val, self.mount_point, img_column=self.img_column, class_column=self.class_column, ga_column=self.ga_column, scalar_column=self.scalar_column, transform=self.valid_transform, repeat_channel=self.repeat_channel)
        self.test_ds = USDataset(self.df_test, self.mount_point, img_column=self.img_column, class_column=self.class_column, ga_column=self.ga_column, scalar_column=self.scalar_column, transform=self.test_transform, repeat_channel=self.repeat_channel)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, shuffle=True, prefetch_factor=2)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last)

class USDataModuleV2(LightningDataModule):
    def __init__(self, df_train, df_val, df_test, mount_point="./", batch_size=256, num_workers=4, img_column="img_path", class_column=None, ga_column=None, scalar_column=None, train_transform=None, valid_transform=None, test_transform=None, drop_last=False, target_column=None):
        super().__init__()

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.mount_point = mount_point
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_column = img_column
        self.class_column = class_column
        self.scalar_column = scalar_column
        self.ga_column = ga_column
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform
        self.drop_last=drop_last    
        self.target_column = target_column

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = USDatasetV2(self.df_train, self.mount_point, img_column=self.img_column, class_column=self.class_column, ga_column=self.ga_column, scalar_column=self.scalar_column, transform=self.train_transform)
        self.val_ds = USDatasetV2(self.df_val, self.mount_point, img_column=self.img_column, class_column=self.class_column, ga_column=self.ga_column, scalar_column=self.scalar_column, transform=self.valid_transform)
        self.test_ds = USDatasetV2(self.df_test, self.mount_point, img_column=self.img_column, class_column=self.class_column, ga_column=self.ga_column, scalar_column=self.scalar_column, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, shuffle=True, prefetch_factor=2)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last)


class SimuDataModule(LightningDataModule):
    def __init__(self, df_train, df_val, df_test, mount_point="./", batch_size=256, num_workers=4, img_column="img_path", class_column=None, ga_column=None, scalar_column=None, train_transform=None, valid_transform=None, test_transform=None, drop_last=False, repeat_channel=True, target_column=None):
        super().__init__()

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.mount_point = mount_point
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_column = img_column
        self.class_column = class_column
        self.scalar_column = scalar_column
        self.ga_column = ga_column
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform
        self.drop_last=drop_last
        self.repeat_channel = repeat_channel
        self.target_column = target_column

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = SimuDataset(self.df_train, self.mount_point, img_column=self.img_column, class_column=self.class_column, ga_column=self.ga_column, scalar_column=self.scalar_column, transform=self.train_transform, repeat_channel=self.repeat_channel, target_column=self.target_column)
        self.val_ds = SimuDataset(self.df_val, self.mount_point, img_column=self.img_column, class_column=self.class_column, ga_column=self.ga_column, scalar_column=self.scalar_column, transform=self.valid_transform, repeat_channel=self.repeat_channel, target_column=self.target_column)
        self.test_ds = SimuDataset(self.df_test, self.mount_point, img_column=self.img_column, class_column=self.class_column, ga_column=self.ga_column, scalar_column=self.scalar_column, transform=self.test_transform, repeat_channel=self.repeat_channel, target_column=self.target_column)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, shuffle=True, prefetch_factor=2)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last)

class USDataModuleBlindSweep(LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()

        # df_train, df_val, df_test, mount_point="./", batch_size=256, num_workers=4, num_frames=50, max_sweeps=-1, img_column='uuid_path', ga_column=None, id_column=None, train_transform=None, valid_transform=None, test_transform=None, drop_last=False

        self.save_hyperparameters(logger=False)

        self.df_train = pd.read_csv(self.hparams.csv_train)
        self.df_val = pd.read_csv(self.hparams.csv_valid)
        self.df_test = pd.read_csv(self.hparams.csv_test)

        
        self.mount_point = self.hparams.mount_point
        self.batch_size = self.hparams.batch_size
        self.num_workers = self.hparams.num_workers
        self.img_column = self.hparams.img_column
        self.ga_column = self.hparams.ga_column
        self.id_column = self.hparams.id_column
        self.num_frames = self.hparams.num_frames
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform
        self.drop_last = self.hparams.drop_last
        self.max_sweeps = self.hparams.max_sweeps

    @staticmethod
    def add_data_specific_args(parent_parser):

        group = parent_parser.add_argument_group("USDataModuleBlindSweep")
        
        group.add_argument('--mount_point', type=str, default="./")
        group.add_argument('--batch_size', type=int, default=8)
        group.add_argument('--num_workers', type=int, default=6)
        group.add_argument('--img_column', type=str, default="img")
        group.add_argument('--ga_column', type=str, default=None)
        group.add_argument('--id_column', type=str, default=None)
        group.add_argument('--csv_train', type=str, default=None, required=True)
        group.add_argument('--csv_valid', type=str, default=None, required=True)
        group.add_argument('--csv_test', type=str, default=None, required=True)
        
        group.add_argument('--drop_last', type=int, default=False)

        return parent_parser

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = USDatasetBlindSweep(self.df_train, mount_point=self.mount_point, num_frames=self.num_frames, img_column=self.img_column, ga_column=self.ga_column,id_column=self.id_column, max_sweeps=self.max_sweeps, transform=self.train_transform)
        self.val_ds = USDatasetBlindSweep(self.df_val, mount_point=self.mount_point, num_frames=self.num_frames, img_column=self.img_column, ga_column=self.ga_column, id_column=self.id_column, max_sweeps=self.max_sweeps, transform=self.valid_transform)
        self.test_ds = USDatasetBlindSweep(self.df_test, mount_point=self.mount_point, num_frames=self.num_frames, img_column=self.img_column, ga_column=self.ga_column, id_column=self.id_column, max_sweeps=self.max_sweeps, transform=self.test_transform)
        

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, shuffle=True, collate_fn=self.pad_seq)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last, collate_fn=self.pad_seq)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last, collate_fn=self.pad_seq)

    def pad_seq(self, batch):

        blind_sweeps = [bs for bs, g in batch]
        ga = [g for v, g in batch]    
        
        blind_sweeps = pad_sequence(blind_sweeps, batch_first=True, padding_value=0.0)
        ga = torch.stack(ga)

        return blind_sweeps, ga
    

class USDataModuleBlindSweepWTag(LightningDataModule):
    def __init__(self, df_train, df_val, df_test, mount_point="./", batch_size=2, num_workers=4, max_sweeps=-1, max_sweeps_val=-1, img_column='uuid_path', ga_column=None, id_column=None, drop_last=False):
        super().__init__()

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.mount_point = mount_point
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_column = img_column
        self.ga_column = ga_column
        self.id_column = id_column


        self.train_transform = ultrasound_transforms.BlindSweepWTagTrainTransforms()
        self.valid_transform = ultrasound_transforms.BlindSweepWTagEvalTransforms()
        self.test_transform = ultrasound_transforms.BlindSweepWTagEvalTransforms()
        self.drop_last=drop_last
        self.max_sweeps = max_sweeps
        self.max_sweeps_val = max_sweeps_val

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = USDatasetBlindSweepWTag(self.df_train, mount_point=self.mount_point, img_column=self.img_column, ga_column=self.ga_column,id_column=self.id_column, max_sweeps=self.max_sweeps, transform=self.train_transform)
        self.val_ds = USDatasetBlindSweepWTag(self.df_val, mount_point=self.mount_point, img_column=self.img_column, ga_column=self.ga_column, id_column=self.id_column, max_sweeps=-1, transform=self.valid_transform)
        self.test_ds = USDatasetBlindSweepWTag(self.df_test, mount_point=self.mount_point, img_column=self.img_column, ga_column=self.ga_column, id_column=self.id_column, max_sweeps=-1, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, shuffle=True, collate_fn=self.pad_seq)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=1, num_workers=self.num_workers, drop_last=self.drop_last, collate_fn=self.pad_seq)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=1, num_workers=self.num_workers, drop_last=self.drop_last, collate_fn=self.pad_seq)

    def pad_seq(self, batch):

        img_d = {}

        keys = batch[0].keys()

        for k in keys:
            if isinstance(k, int):
                img_d[k] = [bs[k].squeeze(0) for bs in batch]
                img_d[k] = pad_sequence(img_d[k], batch_first=True, padding_value=0.0).unsqueeze(1)

        img_d[self.ga_column] = torch.stack([g[self.ga_column] for g in batch])

        if len(img_d[self.ga_column].shape) == 1:
            img_d[self.ga_column] = img_d[self.ga_column].unsqueeze(-1)
        
        img_d['tag'] = torch.stack([g['tag'] for g in batch])

        return img_d


class USDataModuleVolumes(LightningDataModule):
    def __init__(self, df_train, df_val, df_test, mount_point="./", batch_size=32, num_workers=4, max_seq=5, img_column='img_path', ga_column='ga_boe', id_column='study_id', train_transform=None, valid_transform=None, test_transform=None, drop_last=False):
        super().__init__()

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.mount_point = mount_point
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_column = img_column
        self.ga_column = ga_column
        self.id_column= id_column
        self.max_seq = max_seq
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform
        self.drop_last=drop_last

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = USDatasetVolumes(self.df_train, mount_point=self.mount_point, img_column=self.img_column, ga_column=self.ga_column, id_column=self.id_column, max_seq=self.max_seq, transform=self.train_transform)
        self.val_ds = USDatasetVolumes(self.df_val, mount_point=self.mount_point, img_column=self.img_column, ga_column=self.ga_column, id_column=self.id_column, max_seq=self.max_seq, transform=self.valid_transform)
        self.test_ds = USDatasetVolumes(self.df_test, mount_point=self.mount_point, img_column=self.img_column, ga_column=self.ga_column, id_column=self.id_column, max_seq=self.max_seq, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, shuffle=True, collate_fn=self.pad_volumes)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, collate_fn=self.pad_volumes)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last, collate_fn=self.pad_volumes)

    def pad_volumes(self, batch):

        volumes = [v for v, g in batch]
        ga = [g for v, g in batch]    
        
        volumes = pad_sequence(volumes, batch_first=True, padding_value=0.0)
        ga = torch.stack(ga)

        return volumes, ga





class USZDataset(Dataset):
    def __init__(self, df, mount_point = "./", transform=None, img_column="img_path"):
        self.df = df
        self.mount_point = mount_point
        self.transform = transform
        self.img_column = img_column        

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.mount_point, self.df.iloc[idx][self.img_column])

        img_path_z_mu = img_path.replace(".nrrd", "z_mu.nrrd")
        img_path_z_sigma = img_path.replace(".nrrd", "z_mu.nrrd")

        z_mu, head = nrrd.read(img_path_z_mu, index_order="C")
        z_sigma, head = nrrd.read(img_path_z_mu, index_order="C")

        
        z_mu = torch.tensor(z_mu, dtype=torch.float32)
        z_sigma = torch.tensor(z_sigma, dtype=torch.float32)

        if len(z_mu.shape) == 2:
            z_mu = z_mu.unsqueeze(0)
        if len(z_sigma.shape) == 2:
            z_sigma = z_sigma.unsqueeze(0)
            
        
        img = {"z_mu": z_mu, "z_sigma": z_sigma}
        
        if(self.transform):
            img = self.transform(img)

        
        return img

class USZDataModule(LightningDataModule):
    def __init__(self, df_train, df_val, df_test, mount_point="./", batch_size=256, num_workers=4, img_column="img_path", train_transform=None, valid_transform=None, test_transform=None, drop_last=False):
        super().__init__()

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.mount_point = mount_point
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_column = img_column        
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform
        self.drop_last=drop_last        

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = USZDataset(self.df_train, self.mount_point, img_column=self.img_column, transform=self.train_transform)
        self.val_ds = USZDataset(self.df_val, self.mount_point, img_column=self.img_column, transform=self.valid_transform)
        self.test_ds = USZDataset(self.df_test, self.mount_point, img_column=self.img_column, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, shuffle=True, prefetch_factor=2)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last)
    


class FluidDataset(Dataset):
    def __init__(self, df, mount_point = "./", transform=None, img_column="img", seg_column="seg", rois=["Amniotic_fluid", "Maternal_bladder", "Fetal_chest_abdomen", "Fetal_head", "Fetal_heart", "Fetal_limb_other", "Placenta", "Umbilical_cord"]):
        self.df = df
        self.mount_point = mount_point
        self.transform = transform
        self.img_column = img_column        
        self.seg_column = seg_column
        self.rois = rois

        # self.colname_roi_name_map = {'Amniotic_fluid_indicator': 'Amniotic_fluid',
        #                 'Maternal_bladder_indicator': 'Maternal_bladder',
        #                 'fetal_chest_abdomen_indicator': 'Fetal_chest_abdomen',
        #                 'fetal_head_indicator': 'Fetal_head',
        #                 'fetal_heart_indicator': 'Fetal_heart',
        #                 'fetal_limb_other_indicator': 'Fetal_limb_other',
        #                 'placenta_indicator': 'Placenta',
        #                 'umbilical_cord_indicator': 'Umbilical_cord',
        #                 'dropout_indicator': 'Dropout',
        #                 'shadowing_indicator': 'Shadowing'}

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.mount_point, self.df.iloc[idx][self.img_column])
        seg_path = os.path.join(self.mount_point, self.df.iloc[idx][self.seg_column])

        reader = ITKReader()
        img, _ = reader.get_data(reader.read(img_path))

        # img = sitk.ReadImage(img_path)
        img_t = torch.tensor(img)

        reader = ITKReader()
        seg, _ = reader.get_data(reader.read(seg_path))
        seg_t = torch.tensor(seg)
        
        img_d = {self.img_column: img_t, self.seg_column: seg_t}
        
        if self.transform:
            return self.transform(img_d)
        return img_d

class FluidDataModule(LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.df_train = pd.read_csv(self.hparams.csv_train)
        self.df_val = pd.read_csv(self.hparams.csv_valid)
        self.df_test = pd.read_csv(self.hparams.csv_test)
        
        self.mount_point = self.hparams.mount_point
        self.batch_size = self.hparams.batch_size
        self.num_workers = self.hparams.num_workers
        self.img_column = self.hparams.img_column
        self.seg_column = self.hparams.seg_column
        self.drop_last = bool(self.hparams.drop_last)

        self.train_transform = ultrasound_transforms.FluidTrainTransforms()
        self.valid_transform = ultrasound_transforms.FluidEvalTransforms()
        self.test_transform = ultrasound_transforms.FluidEvalTransforms()

    @staticmethod
    def add_data_specific_args(parent_parser):

        group = parent_parser.add_argument_group("SaxiOctreeDataModule")
        
        group.add_argument('--batch_size', type=int, default=8)
        group.add_argument('--num_workers', type=int, default=6)
        group.add_argument('--img_column', type=str, default="img")
        group.add_argument('--seg_column', type=str, default="seg")
        group.add_argument('--csv_train', type=str, default=None, required=True)
        group.add_argument('--csv_valid', type=str, default=None, required=True)
        group.add_argument('--csv_test', type=str, default=None, required=True)
        group.add_argument('--mount_point', type=str, default="./")
        group.add_argument('--drop_last', type=int, default=False)

        return parent_parser

    def collate_fn(self, batch):
        X = [b[self.img_column] for b in batch]

        x_v = [x[0] for x in X]
        x_f = [x[1] for x in X]
        
        X_v = pad_sequence(x_v, batch_first=True, padding_value=0.0)
        X_f = pad_sequence(x_f, batch_first=True, padding_value=0.0)


        Y = [b[self.seg_column] for b in batch]

        y_v = [y[0] for y in Y]
        y_f = [y[1] for y in Y]
        
        Y_v = pad_sequence(y_v, batch_first=True, padding_value=0.0)
        Y_f = pad_sequence(y_f, batch_first=True, padding_value=0.0)
        
        return {self.img_column: (X_v, X_f), self.seg_column: (Y_v, Y_f)}

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = FluidDataset(self.df_train, self.mount_point, img_column=self.img_column, seg_column=self.seg_column, transform=self.train_transform)
        self.val_ds = FluidDataset(self.df_val, self.mount_point, img_column=self.img_column, seg_column=self.seg_column, transform=self.valid_transform)
        self.test_ds = FluidDataset(self.df_test, self.mount_point, img_column=self.img_column, seg_column=self.seg_column, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, shuffle=True, prefetch_factor=2, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, collate_fn=self.collate_fn)


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets, use_max=True):
        self.datasets = datasets
        self.use_max = use_max

    def __getitem__(self, i):
        # print(f"Fetching index: {i}") #debug
        # for d in self.datasets:
            # print(f"Dataset length: {len(d)}") #debug
        return tuple(self.check_len(d, i) for d in self.datasets)

    def __len__(self):
        if self.use_max:
            return max(len(d) for d in self.datasets)
        return min(len(d) for d in self.datasets)

    def shuffle(self):
        for d in self.datasets:
            if isinstance(d, monai.data.Dataset):                
                d.data.df = d.data.df.sample(frac=1.0).reset_index(drop=True)                
            else:
                d.df = d.df.sample(frac=1.0).reset_index(drop=True)

    def check_len(self, d, i):
        if i < len(d):
            return d[i]
        else:
            j = i % len(d)
            return d[j]


class StackDataset(Dataset):
    def __init__(self, dataset, stack_slices=10, shuffle_df=False):        
        self.dataset = dataset
        if shuffle_df:
            self.dataset.df = self.dataset.df.sample(frac=1).reset_index(drop=True)
        self.stack_slices = stack_slices       

    def __len__(self):
        return len(self.dataset)//self.stack_slices

    def __getitem__(self, idx):
        
        start_idx = idx*self.stack_slices

        return torch.stack([self.dataset[idx] for idx in range(start_idx, start_idx + self.stack_slices)], dim=1)

class MRDatasetVolumes(Dataset):
    def __init__(self, df, mount_point = "./", img_column='img_path', id_column='study_id', transform=None):
        self.df = df
        
        self.mount_point = mount_point        
        self.transform = transform
        self.img_column = img_column
        

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.mount_point, self.df.iloc[idx][self.img_column])
        return self.transform(img_path)

class VolumeSlicingProbeParamsDataset(Dataset):
    def __init__(self, diffusor_df: pd.DataFrame, probe_params_df: pd.DataFrame, mount_point="./", transform=None, us_plane_fn='source/blender/simulated_data_export/studies_merged/simulation_ultrasound_plane.stl', n_samples=1000):
        
        self.mount_point = mount_point
        self.transform = transform

        self.df = diffusor_df
        self.probe_params_df = probe_params_df

        self.diffusors = []        

        for _, row in self.df.iterrows():
            diffusor_fn = os.path.join(mount_point, row['img_path'])
            
            diffusor = sitk.ReadImage(os.path.join(self.mount_point, diffusor_fn))
            diffusor_np = sitk.GetArrayFromImage(diffusor)            
            diffusor_t = torch.tensor(diffusor_np.astype(int)).unsqueeze(0)            

            diffusor_spacing = np.array(diffusor.GetSpacing())
            diffusor_size = np.array(diffusor.GetSize())

            diffusor_origin = np.array(diffusor.GetOrigin())
            diffusor_end = diffusor_origin + diffusor_spacing * diffusor_size

            self.diffusors.append({"diffusor_t": diffusor_t, "diffusor_origin": diffusor_origin, "diffusor_end": diffusor_end})

        print("All diffusors loaded!")
            

        self.probe_directions = []
        self.probe_origins = []
        self.n_samples = n_samples

        for _, row in self.probe_params_df.iterrows():
            probe_params = pickle.load(open(os.path.join(self.mount_point, row['probe_param_fn']), 'rb'))
        
            probe_direction = torch.tensor(probe_params['probe_direction'], dtype=torch.float32)
            probe_origin = torch.tensor(probe_params['probe_origin'], dtype=torch.float32)

            self.probe_directions.append(probe_direction.T)
            self.probe_origins.append(probe_origin)

        reader = vtk.vtkSTLReader()
        reader.SetFileName(os.path.join(mount_point, us_plane_fn))
        reader.Update()
        simulation_ultrasound_plane = reader.GetOutput()
        simulation_ultrasound_plane_bounds = np.array(simulation_ultrasound_plane.GetBounds())

        simulation_ultrasound_plane_mesh_grid_size = [256, 256, 1]
        simulation_ultrasound_plane_mesh_grid_params = [torch.arange(start=start, end=end, step=(end - start)/simulation_ultrasound_plane_mesh_grid_size[idx]) for idx, (start, end) in enumerate(zip(simulation_ultrasound_plane_bounds[[0,2,4]], simulation_ultrasound_plane_bounds[[1,3,5]]))]
        self.simulation_ultrasound_plane_mesh_grid = torch.stack(torch.meshgrid(simulation_ultrasound_plane_mesh_grid_params), dim=-1).squeeze().to(torch.float32)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
            
        diffusor_d = random.choice(self.diffusors)
        diffusor_t = diffusor_d['diffusor_t']
        diffusor_origin = diffusor_d['diffusor_origin']
        diffusor_end = diffusor_d['diffusor_end']

        is_empty = True

        while is_empty:

            ridx = random.randint(0, len(self.probe_directions)-1)
            probe_direction = self.probe_directions[ridx]
            probe_origin = self.probe_origins[ridx]

            simulation_ultrasound_plane_mesh_grid_transformed_t = self.transform_simulation_ultrasound_plane_single(probe_direction, probe_origin, diffusor_origin, diffusor_end)

            diffusor_sample_t = self.diffusor_sampling(diffusor_t.unsqueeze(0).to(torch.float32), simulation_ultrasound_plane_mesh_grid_transformed_t.unsqueeze(0)).squeeze().unsqueeze(0)
            
            if torch.any(torch.isin(diffusor_sample_t[:, 128:, 64:192], torch.tensor([4, 7]))):
                is_empty = False

        if self.transform:
            diffusor_sample_t = self.transform(diffusor_sample_t)
        
        return diffusor_sample_t

    def transform_simulation_ultrasound_plane(self, probe_directions, probe_origins, diffusor_origin, diffusor_end):

        simulation_ultrasound_plane_mesh_grid_transformed_t = []
        for probe_origin, probe_direction in zip(probe_origins, probe_directions):

            simulation_ultrasound_plane_mesh_grid_transformed = self.transform_simulation_ultrasound_plane_single(probe_direction, probe_origin, diffusor_origin, diffusor_end)

            simulation_ultrasound_plane_mesh_grid_transformed_t.append(simulation_ultrasound_plane_mesh_grid_transformed)
        
        simulation_ultrasound_plane_mesh_grid_transformed_t = torch.cat(simulation_ultrasound_plane_mesh_grid_transformed_t, dim=0)
        
        return simulation_ultrasound_plane_mesh_grid_transformed_t
    
    def transform_simulation_ultrasound_plane_single(self, probe_direction, probe_origin, diffusor_origin, diffusor_end):

        theta = torch.rand(3) * torch.tensor([0.05*torch.pi, 0.05*torch.pi, 2*torch.pi])
        # Generate rotation matrix
        R = self.euler_angles_to_rotation_matrix(theta)
        probe_direction_rotated = torch.matmul(probe_direction, R)
        probe_origin_translated = torch.randn(3)*0.001 + probe_origin

        simulation_ultrasound_plane_mesh_grid_transformed = torch.matmul(self.simulation_ultrasound_plane_mesh_grid, probe_direction_rotated) + probe_origin_translated
        simulation_ultrasound_plane_mesh_grid_transformed = 2.*(simulation_ultrasound_plane_mesh_grid_transformed - diffusor_origin)/(diffusor_end - diffusor_origin) - 1.

        return simulation_ultrasound_plane_mesh_grid_transformed.to(torch.float32).unsqueeze(0)
    
    def diffusor_sampling(self, diffusor_t, simulation_ultrasound_plane_mesh_grid_transformed_t):
        return F.grid_sample(diffusor_t, simulation_ultrasound_plane_mesh_grid_transformed_t, mode='nearest', align_corners=True)
    
    def read_probe_params(self, probe_params_fn):
        return pickle.load(open(os.path.join(self.mount_point, probe_params_fn), 'rb'))

    # Function to create a rotation matrix from Euler angles
    def euler_angles_to_rotation_matrix(self, theta):
        R_x = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(theta[0]), -torch.sin(theta[0])],
            [0, torch.sin(theta[0]), torch.cos(theta[0])]
        ])

        R_y = torch.tensor([
            [torch.cos(theta[1]), 0, torch.sin(theta[1])],
            [0, 1, 0],
            [-torch.sin(theta[1]), 0, torch.cos(theta[1])]
        ])

        R_z = torch.tensor([
            [torch.cos(theta[2]), -torch.sin(theta[2]), 0],
            [torch.sin(theta[2]), torch.cos(theta[2]), 0],
            [0, 0, 1]
        ])

        R = torch.mm(torch.mm(R_z, R_y), R_x)
        return R


class VolumeSlicingDataModule(LightningDataModule):
    def __init__(self, df_train, df_val, df_test, df_probe_params, mount_point="./", batch_size=256, num_workers=1, img_column="img_path", seg_column="seg_path", train_transform=None, valid_transform=None, test_transform=None, drop_last=False, n_samples_train=1000, n_samples_val=100):
        super().__init__()

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.df_probe_params = df_probe_params
        self.n_samples_train = n_samples_train
        self.n_samples_val = n_samples_val
        self.mount_point = mount_point
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_column = img_column        
        self.seg_column = seg_column        
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform
        self.drop_last=drop_last        

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = monai.data.Dataset(data=VolumeSlicingProbeParamsDataset(self.df_train, probe_params_df=self.df_probe_params, mount_point=self.mount_point, transform=self.train_transform, n_samples=self.n_samples_train))
        self.val_ds = monai.data.Dataset(data=VolumeSlicingProbeParamsDataset(self.df_val, probe_params_df=self.df_probe_params, mount_point=self.mount_point, transform=self.train_transform, n_samples=self.n_samples_val))
        self.test_ds = monai.data.Dataset(data=VolumeSlicingProbeParamsDataset(self.df_test, probe_params_df=self.df_probe_params, mount_point=self.mount_point, transform=self.train_transform))

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last, persistent_workers=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last)

class MRDataModuleVolumes(LightningDataModule):
    def __init__(self, df_train, df_val, df_test, mount_point="./", batch_size=32, num_workers=4, img_column='img_path', ga_column='ga_boe', id_column='study_id', train_transform=None, valid_transform=None, test_transform=None, drop_last=False):
        super().__init__()

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.mount_point = mount_point
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_column = img_column
        self.ga_column = ga_column
        self.id_column= id_column
        self.max_seq = max_seq
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform
        self.drop_last=drop_last

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = MRDatasetVolumes(self.df_train, mount_point=self.mount_point, img_column=self.img_column, id_column=self.id_column, transform=self.train_transform)
        self.val_ds = MRDatasetVolumes(self.df_val, mount_point=self.mount_point, img_column=self.img_column, id_column=self.id_column, transform=self.valid_transform)
        self.test_ds = MRDatasetVolumes(self.df_test, mount_point=self.mount_point, img_column=self.img_column, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, shuffle=True, collate_fn=self.arrange_slices)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, collate_fn=self.arrange_slices)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last, collate_fn=self.arrange_slices)

    def arrange_slices(self, batch):
        batch = torch.cat(batch, axis=1).permute(dims=(1,0,2,3))
        print(f"Batch shape after concatenation and permutation: {batch.shape}") #debug
        idx = torch.randperm(batch.shape[0])
        return batch[idx]

class MRUSDataModule(LightningDataModule):
    def __init__(self, mr_dataset_train, mr_dataset_val, us_dataset_train, us_dataset_val, batch_size=4, num_workers=4):
        super().__init__()

        self.mr_dataset_train = mr_dataset_train
        self.us_dataset_train = us_dataset_train

        self.mr_dataset_val = mr_dataset_val
        self.us_dataset_val = us_dataset_val
        
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = ConcatDataset(self.mr_dataset_train, self.us_dataset_train)
        self.val_ds = ConcatDataset(self.mr_dataset_val, self.us_dataset_val)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, shuffle=True, collate_fn=self.arrange_slices)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, collate_fn=self.arrange_slices)    

    def arrange_slices(self, batch):
        mr_batch = [mr for mr, us in batch]
        us_batch = [us for mr, us in batch]        
        mr_batch = torch.cat(mr_batch, axis=1).permute(dims=(1,0,2,3))
        us_batch = torch.cat(us_batch, axis=1).permute(dims=(1,0,2,3))        
        return mr_batch[torch.randperm(mr_batch.shape[0])], us_batch

class ConcatDataModule(LightningDataModule):
    def __init__(self, datasetA_train, datasetA_val, datasetB_train, datasetB_val, batch_size=8, num_workers=4):
        super().__init__()

        self.datasetA_train = datasetA_train
        self.datasetB_train = datasetB_train

        self.datasetA_val = datasetA_val
        self.datasetB_val = datasetB_val
        
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders        
        self.train_ds = ConcatDataset(self.datasetA_train, self.datasetB_train)
        self.val_ds = ConcatDataset(self.datasetA_val, self.datasetB_val)

    def train_dataloader(self):
        self.train_ds.shuffle()
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)    
        

class DiffusorDataset(Dataset):
    def __init__(self, df, mount_point = "./", img_column='img_path', pc_column=None, transform=None):
        self.df = df
        
        self.mount_point = mount_point        
        self.transform = transform
        self.img_column = img_column
        self.pc_column = pc_column

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.mount_point, self.df.iloc[idx][self.img_column])

        diffusor_np, diffusor_head = nrrd.read(img_path)
        diffusor_t = torch.tensor(diffusor_np.astype(int)).permute(2, 1, 0)
        diffusor_size = torch.tensor(diffusor_head['sizes'])
        diffusor_spacing = torch.tensor(np.diag(diffusor_head['space directions']))

        diffusor_origin = torch.tensor(diffusor_head['space origin']).flip(dims=[0])
        diffusor_end = diffusor_origin + diffusor_spacing * diffusor_size

        if self.transform:
            return self.transform(diffusor_t), diffusor_origin, diffusor_end
        return diffusor_t, diffusor_origin, diffusor_end    
    
class DiffusorDataModule(LightningDataModule):
    def __init__(self, df_train, df_val, mount_point="./", batch_size=256, num_workers=4, img_column="img_path", train_transform=None, valid_transform=None, drop_last=False):
        super().__init__()

        self.df_train = df_train
        self.df_val = df_val
        self.mount_point = mount_point
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_column = img_column
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.drop_last = drop_last

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = DiffusorDataset(self.df_train, self.mount_point, img_column=self.img_column, transform=self.train_transform)
        self.val_ds = DiffusorDataset(self.df_val, self.mount_point, img_column=self.img_column, transform=self.valid_transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, shuffle=True, prefetch_factor=2)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last)

class DiffusorSampleDataset(Dataset):
    def __init__(self, df, mount_point = "./", img_column='img_path', transform=None, num_samples=None, return_ridx=False):
        self.df = df
        
        self.mount_point = mount_point        
        self.transform = transform
        self.img_column = img_column
        self.use_random_samples = False
        if num_samples is None:
            self.num_samples = len(self.df.index)
        else:
            self.num_samples = num_samples
            self.use_random_samples = True
        self.return_ridx = return_ridx        

        self.buffer = []
        for idx in range(len(self.df.index)):
            img_path = os.path.join(self.mount_point, self.df.iloc[idx][self.img_column])

            diffusor = sitk.ReadImage(img_path)
            diffusor_t = torch.tensor(sitk.GetArrayFromImage(diffusor).astype(int))
            diffusor_size = torch.tensor(diffusor.GetSize())
            diffusor_spacing = torch.tensor(diffusor.GetSpacing())
            diffusor_origin = torch.tensor(diffusor.GetOrigin())
            diffusor_end = diffusor_origin + diffusor_spacing * diffusor_size

            self.buffer.append([diffusor_t, diffusor_origin, diffusor_end])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.use_random_samples:
            idx = random.randint(0, len(self.buffer)-1)
        diffusor_t, diffusor_origin, diffusor_end = self.buffer[idx] 
        if self.transform:
            diffusor_t = self.transform(diffusor_t)            
        if self.return_ridx:
            return diffusor_t, diffusor_origin, diffusor_end, torch.tensor(idx)
        return diffusor_t, diffusor_origin, diffusor_end
    
class DiffusorSampleDataModule(LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.df_train = pd.read_csv(self.hparams.csv_train)
        self.df_val = pd.read_csv(self.hparams.csv_valid)
        self.df_test = pd.read_csv(self.hparams.csv_test)

    @staticmethod
    def add_data_specific_args(parent_parser):

        group = parent_parser.add_argument_group("DiffusorSampleDataModule")
        
        group.add_argument('--batch_size', type=int, default=8)
        group.add_argument('--num_workers', type=int, default=6)
        group.add_argument('--img_column', type=str, default="img")
        group.add_argument('--num_samples_train', type=int, default=1000)
        group.add_argument('--num_samples_val', type=int, default=100)
        group.add_argument('--csv_train', type=str, default=None, required=True)
        group.add_argument('--csv_valid', type=str, default=None, required=True)
        group.add_argument('--mount_point', type=str, default="./")
        group.add_argument('--drop_last', type=int, default=0)        

        return parent_parser

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = DiffusorSampleDataset(self.df_train, self.hparams.mount_point, img_column=self.hparams.img_column, num_samples=self.hparams.num_samples_train)
        self.val_ds = DiffusorSampleDataset(self.df_val, self.hparams.mount_point, img_column=self.hparams.img_column, num_samples=self.hparams.num_samples_val)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last, shuffle=True, prefetch_factor=2)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last)
    

class DiffusorSampleSurfDataset(Dataset):
    def __init__(self, df, mount_point = "./", img_column='img_path', surf_column=None, surf_idx_fn=None, transform=None, num_samples=1000):
        self.df = df
        
        self.mount_point = mount_point        
        self.transform = transform
        self.img_column = img_column
        self.num_samples = num_samples
        self.return_ridx = return_ridx
        self.surf_column = surf_column
        self.surf_idx_fn = surf_idx_f

        surf_idx = None
        if self.surf_idx_fn:
            surf_idx = np.load(self.surf_idx_fn)
            surf_idx = torch.tensor(idx, dtype=torch.int64, device=V.device)

        self.buffer = []
        for idx in range(len(self.df.index)):
            img_path = os.path.join(self.mount_point, self.df.iloc[idx][self.img_column])

            diffusor_np, diffusor_head = nrrd.read(img_path)            
            diffusor_size = diffusor_head['sizes']
            diffusor_spacing = np.diag(diffusor_head['space directions'])

            diffusor_origin = np.flip(diffusor_head['space origin'], axis=0)
            diffusor_end = diffusor_origin + diffusor_spacing * diffusor_size
            
            surf_path = os.path.join(self.mount_point, self.df.iloc[idx][self.surf_column])
            surf = utils.ReadSurf(surf_path)
            verts, faces = utils.PolyDataToTensors_v_f(surf)

            if surf_idx:
                verts = knn_gather(verts.unsqueeze(0), surf_idx).squeeze(-2).squeeze(0).contiguous()
                self.buffer.append((diffusor_np, diffusor_origin, diffusor_end, verts))
            else:
                self.buffer.append((diffusor_np, diffusor_origin, diffusor_end, verts, faces))


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        r_idx = random.randint(0, len(self.buffer)-1)
        diffusor_np, diffusor_origin, diffusor_end, V, F = self.buffer[r_idx] 
        diffusor_t = torch.tensor(diffusor_np.copy().astype(int)).permute(2, 1, 0)     
        if self.transform:
            diffusor_t = self.transform(diffusor_t)            
        if self.return_ridx:
            return diffusor_t, torch.tensor(diffusor_origin.copy()), torch.tensor(diffusor_end.copy()), torch.tensor(r_idx)
        return diffusor_t, torch.tensor(diffusor_origin.copy()), torch.tensor(diffusor_end.copy()), V, F
    
class DiffusorSampleSurfDataModule(LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.df_train = pd.read_csv(self.hparams.csv_train)
        self.df_val = pd.read_csv(self.hparams.csv_valid)

        self.train_transform = None
        self.valid_transform = None

    @staticmethod
    def add_data_specific_args(parent_parser):

        group = parent_parser.add_argument_group("DiffusorSampleSurfDataModule")
        
        group.add_argument('--batch_size', type=int, default=8)
        group.add_argument('--num_workers', type=int, default=6)
        group.add_argument('--img_column', type=str, default="img")
        group.add_argument('--surf_column', type=str, default="surf")
        group.add_argument('--surf_idx_fn', type=str, default=None)
        group.add_argument('--num_samples_train', type=int, default=1000)
        group.add_argument('--num_samples_val', type=int, default=100)
        group.add_argument('--csv_train', type=str, default=None, required=True)
        group.add_argument('--csv_valid', type=str, default=None, required=True)
        group.add_argument('--mount_point', type=str, default="./")
        group.add_argument('--drop_last', type=int, default=0)
        
        

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = DiffusorSampleSurfDataset(self.df_train, self.hparams.mount_point, img_column=self.hparams.img_column, surf_column=self.hparams.surf_column, surf_idx_fn=self.hparams.surf_idx_fn, transform=self.train_transform, num_samples=self.num_samples_train)
        self.val_ds = DiffusorSampleSurfDataset(self.df_val, self.hparams.mount_point, img_column=self.hparams.img_column, surf_column=self.hparams.surf_column, surf_idx_fn=self.hparams.surf_idx_fn, transform=self.valid_transform, num_samples=self.num_samples_val)

    def pad_verts_faces(self, batch):
        # Collate function for the dataloader to know how to combine the data

        if self.hparams.surf_idx_fn:

            diffusor = [d for d, do, de, v  in batch]
            diffusor_origin = [do for d, do, de, v in batch]
            diffusor_end = [de for d, do, de, v in batch]
            
            verts = [v for d, do, de, v  in batch]
                
            return torch.stack(diffusor), torch.stack(diffusor_origin), torch.stack(diffusor_end), torch.stack(verts)

        else:
            diffusor = [d for d, do, de, v, f  in batch]
            diffusor_origin = [do for d, do, de, v, f in batch]
            diffusor_end = [de for d, do, de, v, f in batch]
            
            verts = [v for d, do, de, v, f  in batch]
            faces = [f for d, do, de, v, f in batch]
            
            verts = pad_sequence(verts, batch_first=True, padding_value=0.0)        
            faces = pad_sequence(faces, batch_first=True, padding_value=-1)
                
            return torch.stack(diffusor), torch.stack(diffusor_origin), torch.stack(diffusor_end), verts, faces

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last, shuffle=True, prefetch_factor=2, collate_fn=self.pad_verts_faces)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last, collate_fn=self.pad_verts_faces)
    


class RealUSSurfDataset(Dataset):
    def __init__(self, df, mount_point = "./", img_column='img_path', surf_column='surf_path', transform=None, num_samples=1000, return_ridx=False):
        self.df = df
        
        self.mount_point = mount_point        
        self.transform = transform
        self.img_column = img_column
        self.num_samples = num_samples
        self.return_ridx = return_ridx
        self.surf_column = surf_column

        self.buffer = []
        for idx in range(len(self.df.index)):
            img_path = os.path.join(self.mount_point, self.df.iloc[idx][self.img_column])

            diffusor_np, diffusor_head = nrrd.read(img_path)            
            diffusor_size = diffusor_head['sizes']
            diffusor_spacing = np.diag(diffusor_head['space directions'])

            diffusor_origin = np.flip(diffusor_head['space origin'], axis=0)
            diffusor_end = diffusor_origin + diffusor_spacing * diffusor_size
            
            surf_path = os.path.join(self.mount_point, self.df.iloc[idx][self.surf_column])
            surf = utils.ReadSurf(surf_path)
            verts, faces = utils.PolyDataToTensors_v_f(surf)

            self.buffer.append((diffusor_np, diffusor_origin, diffusor_end, verts, faces))

        self.tags_dict = {'M': 0, 'L0': 1, 'L1': 2, 'R0': 3, 'R1': 4, 'C1': 5, 'C2': 6, 'C3': 7, 'C4': 8}


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        r_idx = random.randint(0, len(self.buffer)-1)
        diffusor_np, diffusor_origin, diffusor_end, V, F = self.buffer[r_idx] 
        diffusor_t = torch.tensor(diffusor_np.copy().astype(int)).permute(2, 1, 0)     
        if self.transform:
            diffusor_t = self.transform(diffusor_t)            
        if self.return_ridx:
            return diffusor_t, torch.tensor(diffusor_origin.copy()), torch.tensor(diffusor_end.copy()), torch.tensor(r_idx)
        return diffusor_t, torch.tensor(diffusor_origin.copy()), torch.tensor(diffusor_end.copy()), V, F
    
class RealUSSurfDataModule(LightningDataModule):
    def __init__(self, df_train, df_val, mount_point="./", batch_size=256, num_workers=4, img_column="img_path", surf_column="surf_path", train_transform=None, valid_transform=None, drop_last=False, num_samples_train=1000, num_samples_val=100):
        super().__init__()

        self.df_train = df_train
        self.df_val = df_val
        self.mount_point = mount_point
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_column = img_column
        self.surf_column = surf_column
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.drop_last = drop_last
        self.num_samples_train = num_samples_train
        self.num_samples_val = num_samples_val

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = RealUSSurfDataset(self.df_train, self.mount_point, img_column=self.img_column, surf_column=self.surf_column, transform=self.train_transform, num_samples=self.num_samples_train)
        self.val_ds = RealUSSurfDataModule(self.df_val, self.mount_point, img_column=self.img_column, surf_column=self.surf_column, transform=self.valid_transform, num_samples=self.num_samples_val)

    def pad_verts_faces(self, batch):
        # Collate function for the dataloader to know how to comine the data

        diffusor = [d for d, do, de, v, f  in batch]
        diffusor_origin = [do for d, do, de, v, f in batch]
        diffusor_end = [de for d, do, de, v, f in batch]
        
        verts = [v for d, do, de, v, f  in batch]
        faces = [f for d, do, de, v, f in batch]
        
        verts = pad_sequence(verts, batch_first=True, padding_value=0.0)        
        faces = pad_sequence(faces, batch_first=True, padding_value=-1)
            
        return torch.stack(diffusor), torch.stack(diffusor_origin), torch.stack(diffusor_end), verts, faces

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last, shuffle=True, prefetch_factor=2, collate_fn=self.pad_verts_faces)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last, collate_fn=self.pad_verts_faces)

class ImgPCDataset(Dataset):
    def __init__(self, df_path, pc_path, mount_point="./", transform=None, allow_pickle=False, num_samples=None):
        super().__init__()        
        self.transform = transform
        self.pointclouds = np.load(pc_path, allow_pickle=False)
        self.num_samples = num_samples
        self.df = pd.read_csv(df_path)

        self.img = []

        for idx, row in self.df.iterrows():
            img_fn = self.df.iloc[idx]['img']
            img_fn = os.path.join(mount_point, img_fn)
            img_np, img_head = nrrd.read(img_fn)
            img_t = torch.tensor(img_np.astype(int)).permute(2, 1, 0).unsqueeze(0).float()

            img_size = torch.tensor(img_head['sizes'])
            img_spacing = torch.tensor(np.diag(img_head['space directions']))

            img_origin = torch.tensor(img_head['space origin']).flip(dims=[0])
            img_end = img_origin + img_spacing * img_size
            
            self.img.append((img_t, img_origin, img_end))


    def __len__(self):
        if self.num_samples:
            return self.num_samples
        return len(self.pointclouds)

    def __getitem__(self, idx):

        if self.num_samples:
            idx = random.randint(0, len(self.pointclouds) - 1)

        pc = torch.tensor(self.pointclouds[idx]).to(torch.float32)
        if self.transform is not None:
            pc = self.transform(pc).to(torch.float32)

        img_t, img_origin, img_end = self.img[idx]

        return img_t, img_origin, img_end, pc
    
class ImgPCDataModule(LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.train_transform = None
        self.valid_transform = None
        self.test_transform = None

        if self.hparams.rescale_factor is not None:
            self.hparams.rescale_factor = 1.0
            self.train_transform = saxi_transforms.EvalTransform(rescale_factor=self.hparams.rescale_factor)
            self.valid_transform = saxi_transforms.EvalTransform(rescale_factor=self.hparams.rescale_factor)
            self.test_transform = saxi_transforms.EvalTransform(rescale_factor=self.hparams.rescale_factor)
        
    @staticmethod
    def add_data_specific_args(parent_parser):

        group = parent_parser.add_argument_group("Loads a point cloud from a numpy file and also an images listed in a csv file")
        
        # Datasets and loaders
        group.add_argument('--mount_point', type=str, default="./", help="Mount point for the data")
        group.add_argument('--csv_train', type=str, required=True, help="Path to the csv file containing the image paths. Must contain column img")
        group.add_argument('--np_train', type=str, required=True, help="Path to the numpy file containing the point clouds. The order of the numpy array must match the order of the images in the csv file, shape is (N, P, 3)")
        group.add_argument('--num_samples_train', type=int, default=None, help="Number of samples to use during training. The loading repeats the point clouds and images if the number of samples is less than the number of point clouds")
        group.add_argument('--csv_valid', type=str, required=True, help="Path to the csv file containing the image paths")
        group.add_argument('--np_valid', type=str, required=True, help="Path to the numpy file containing the point clouds")
        group.add_argument('--csv_test', type=str, required=True, help="Path to the csv file containing the image paths")
        group.add_argument('--np_test', type=str, required=True, help="Path to the numpy file containing the point clouds")
        group.add_argument('--batch_size', type=int, default=4, help="Batch size for the train dataloaders")
        group.add_argument('--num_workers', type=int, default=1)
        group.add_argument('--rescale_factor', type=float, default=None)

        return parent_parser
        

    def setup(self,stage=None):
        # Assign train/val datasets for use in dataloaders
        self.train_dataset = ImgPCDataset(self.hparams.csv_train, self.hparams.np_train, mount_point=self.hparams.mount_point, num_samples=self.hparams.num_samples_train, transform=self.train_transform)
        self.val_dataset = ImgPCDataset(self.hparams.csv_valid, self.hparams.np_valid, mount_point=self.hparams.mount_point, transform=self.valid_transform)
        self.test_dataset = ImgPCDataset(self.hparams.csv_test, self.hparams.np_test, mount_point=self.hparams.mount_point, transform=self.test_transform)

    def train_dataloader(self):  
        return DataLoader(self.train_dataset,batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=1, drop_last=True)

class USCutDataset(Dataset):
    def __init__(self, dfs, mount_point = "./", transform=None, img_column="img_path", class_column="pred_class", num_samples=1000):
        self.dfs = dfs
        self.mount_point = mount_point
        self.transform = transform
        self.img_column = img_column
        self.class_column = class_column
        self.num_samples = num_samples

    # def shuffle(self):
    #     """
    #     Shuffle the rows of multiple aligned dataframes in two steps:
    #     1. Within each dataframe, group rows by class (using 'class_pred'),
    #         shuffle the rows in each class (with the same order across all dfs),
    #         and then reassemble the dataframe in the order specified by unique_classes.
    #     2. Generate one overall permutation and apply it identically to all dataframes.
        
    #     This ensures that, after shuffling, the i-th row in every dataframe belongs to the same class.
        
    #     Returns:
    #         list of pd.DataFrame: The list of shuffled dataframes.
    #     """
    #     # Step 1: For each dataframe, group rows by class, shuffle them, and then reassemble the dataframe.

    #     unique_classes = self.dfs[0][self.class_column].unique()
    #     random.shuffle(unique_classes)

    #     shuffled_dfs = []
        
    #     for df in self.dfs:
    #         # Group indices by class.            
    #         shuffled_df = []
    #         for cls in unique_classes:
    #             group = df.query(f"{self.class_column} == {cls}").sample(frac=1.0)
    #             shuffled_df.append(group)
    #         # Concatenate the shuffled groups.
    #         shuffled_df = pd.concat(shuffled_df, ignore_index=True).reset_index(drop=True)
    #         shuffled_dfs.append(shuffled_df)

    #     # Step 2: Generate one overall permutation for the full dataframe.
    #     total_rows = len(shuffled_dfs[0])
    #     perm = list(range(total_rows))
    #     random.shuffle(perm)
        
    #     # Apply the same permutation to every dataframe.
    #     shuffled_dfs = [df.iloc[perm].reset_index(drop=True) for df in shuffled_dfs]
        
    #     self.dfs = shuffled_dfs

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        imgs_t = []
        
        img_source_arr = []
        label_source_arr = []
        img_target_arr = []
        label_target_arr = []
        
        for idx_s, df in enumerate(self.dfs):
            img_path = os.path.join(self.mount_point, df.sample(n=1)[self.img_column].values[0])
            img = sitk.ReadImage(img_path)
            img_source_t = torch.tensor(sitk.GetArrayFromImage(img), dtype=torch.float32)
            if img.GetNumberOfComponentsPerPixel() == 1:
                img_source_t = img_source_t.unsqueeze(-1)
            else:
                img_source_t = img_source_t[:,:,:1]
            img_source_t = img_source_t.permute(2, 0, 1)  # Change to (C, H, W)
            if self.transform:
                img_source_t = self.transform(img_source_t)
            img_source_arr.append(img_source_t)
            label_source_arr.append(torch.tensor(idx_s))

            idx_t = random.randint(0, len(self.dfs) - 2)
            # If idx_t is greater than or equal to idx_s (current df) the excluded number, shift it up by one. That ensure that it picks a different df.
            if idx_t >= idx_s:
                idx_t += 1

            img_path = os.path.join(self.mount_point, self.dfs[idx_t].sample(n=1)[self.img_column].values[0])
            img = sitk.ReadImage(img_path)
            img_target_t = torch.tensor(sitk.GetArrayFromImage(img), dtype=torch.float32)
            if img.GetNumberOfComponentsPerPixel() == 1:
                img_target_t = img_target_t.unsqueeze(-1)
            else:
                img_target_t = img_target_t[:,:,:1]
            img_target_t = img_target_t.permute(2, 0, 1)  # Change to (C, H, W)
            if self.transform:
                img_target_t = self.transform(img_target_t)
            img_target_arr.append(img_target_t)
            label_target_arr.append(torch.tensor(idx_t))

        return img_source_arr, label_source_arr, img_target_arr, label_target_arr
        

class CutDataModule(LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.dfs_train = [pd.read_parquet(csv) for csv in self.hparams.csv_train]
        self.dfs_valid = [pd.read_parquet(csv) for csv in self.hparams.csv_valid]
        self.dfs_test = [pd.read_parquet(csv) for csv in self.hparams.csv_test]

        self.train_transform = ultrasound_transforms.USCutTrainTransforms()
        self.test_transform = ultrasound_transforms.USClassEvalTransforms()


    @staticmethod
    def add_data_specific_args(parent_parser):

        group = parent_parser.add_argument_group("Loads a series of images to train a ClassConditionedCut Model")
        
        # Datasets and loaders
        group.add_argument('--mount_point', type=str, default="./", help="Mount point for the data")
        group.add_argument('--csv_train', type=str, action='append', required=True, help="Path to the parquet file containing the image paths. Must contain column file_path and class_pred")
        group.add_argument('--csv_valid', type=str, action='append', required=True, help="Path to the parquet file containing the image paths")
        group.add_argument('--csv_test', type=str, action='append', required=True, help="Path to the parquet file containing the image paths")
        group.add_argument('--img_column', type=str, default="file_path")
        group.add_argument('--class_column', type=str, default="class_pred")
        group.add_argument('--batch_size', type=int, default=32, help="Batch size for the train dataloaders")
        group.add_argument('--num_workers', type=int, default=1)
        group.add_argument('--prefetch_factor', type=int, default=2)
        group.add_argument('--drop_last', type=int, default=False)
        group.add_argument('--num_samples_train', type=int, default=10000)
        group.add_argument('--num_samples_val', type=int, default=1000)
        group.add_argument('--num_samples_test', type=int, default=100)

        return parent_parser        

    def collate_all(self, batch):
        
        source_img_t = torch.cat([torch.stack(s_i) for s_i, s_l, t_i, t_l in batch])
        source_label_t = torch.cat([torch.stack(s_l) for s_i, s_l, t_i, t_l in batch])
        target_img_t = torch.cat([torch.stack(t_i) for s_i, s_l, t_i, t_l in batch])
        target_label_t = torch.cat([torch.stack(t_l) for s_i, s_l, t_i, t_l in batch])
        
        return source_img_t,source_label_t, target_img_t, target_label_t

    def setup(self, stage=None):
        self.train_ds = USCutDataset(self.dfs_train, self.hparams.mount_point, img_column=self.hparams.img_column, num_samples=self.hparams.num_samples_train, transform=self.train_transform)
        self.val_ds = USCutDataset(self.dfs_valid, self.hparams.mount_point, img_column=self.hparams.img_column, num_samples=self.hparams.num_samples_val, transform=self.test_transform)
        self.test_ds = USCutDataset(self.dfs_test, self.hparams.mount_point, img_column=self.hparams.img_column, num_samples=self.hparams.num_samples_test, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.hparams.drop_last, shuffle=True, prefetch_factor=self.hparams.prefetch_factor, collate_fn=self.collate_all)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, drop_last=self.hparams.drop_last, collate_fn=self.collate_all)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=1, num_workers=self.hparams.num_workers, collate_fn=self.collate_all)

class USCutDatasetV2(Dataset):
    def __init__(self, dfs, mount_point = "./", transform=None, img_column="img_path", class_column="pred_class", num_samples=1000):
        self.dfs = dfs
        self.mount_point = mount_point
        self.transform = transform
        self.img_column = img_column
        self.class_column = class_column
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        imgs_t = []
        
        img_source_arr = []
        for df in self.dfs[0:-1]:
            img_path = os.path.join(self.mount_point, df.sample(n=1)[self.img_column].values[0])
            img = sitk.ReadImage(img_path)
            img_source_t = torch.tensor(sitk.GetArrayFromImage(img), dtype=torch.float32)
            if img.GetNumberOfComponentsPerPixel() == 1:
                img_source_t = img_source_t.unsqueeze(-1)
            else:
                img_source_t = img_source_t[:,:,:1]
            img_source_t = img_source_t.permute(2, 0, 1)  # Change to (C, H, W)
            if self.transform:
                img_source_t = self.transform(img_source_t)
            img_source_arr.append(img_source_t)

        img_target_arr = []
        df_target = self.dfs[-1]
        for i in range(len(img_source_arr)):
            img_path = os.path.join(self.mount_point, df_target.sample(n=1)[self.img_column].values[0])
            img = sitk.ReadImage(img_path)
            img_target_t = torch.tensor(sitk.GetArrayFromImage(img), dtype=torch.float32)
            if img.GetNumberOfComponentsPerPixel() == 1:
                img_target_t = img_target_t.unsqueeze(-1)
            else:
                img_target_t = img_target_t[:,:,:1]
            img_target_t = img_target_t.permute(2, 0, 1)  # Change to (C, H, W)
            if self.transform:
                img_target_t = self.transform(img_target_t)
            img_target_arr.append(img_target_t)
        
        return img_source_arr, img_target_arr

class CutDataModuleV2(LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.dfs_train = [pd.read_parquet(csv) for csv in self.hparams.csv_train]
        self.dfs_valid = [pd.read_parquet(csv) for csv in self.hparams.csv_valid]
        self.dfs_test = [pd.read_parquet(csv) for csv in self.hparams.csv_test]

        self.train_transform = ultrasound_transforms.USCutTrainTransforms()
        self.test_transform = ultrasound_transforms.USClassEvalTransforms()


    @staticmethod
    def add_data_specific_args(parent_parser):

        group = parent_parser.add_argument_group("Loads a series of images to train a ClassConditionedCut Model")
        
        # Datasets and loaders
        group.add_argument('--mount_point', type=str, default="./", help="Mount point for the data")
        group.add_argument('--csv_train', type=str, action='append', required=True, help="Path to the parquet file containing the image paths. Must contain column file_path and class_pred")
        group.add_argument('--csv_valid', type=str, action='append', required=True, help="Path to the parquet file containing the image paths")
        group.add_argument('--csv_test', type=str, action='append', required=True, help="Path to the parquet file containing the image paths")
        group.add_argument('--img_column', type=str, default="file_path")
        group.add_argument('--class_column', type=str, default="class_pred")
        group.add_argument('--batch_size', type=int, default=32, help="Batch size for the train dataloaders")
        group.add_argument('--num_workers', type=int, default=1)
        group.add_argument('--prefetch_factor', type=int, default=2)
        group.add_argument('--drop_last', type=int, default=False)
        group.add_argument('--num_samples_train', type=int, default=10000)
        group.add_argument('--num_samples_val', type=int, default=1000)
        group.add_argument('--num_samples_test', type=int, default=100)

        return parent_parser        

    def collate_all(self, batch):
        
        source_t = torch.cat([torch.stack(s) for s, t in batch])
        target_t = torch.cat([torch.stack(t) for s, t in batch])
        
        return source_t, target_t

    def setup(self, stage=None):
        self.train_ds = USCutDatasetV2(self.dfs_train, self.hparams.mount_point, img_column=self.hparams.img_column, num_samples=self.hparams.num_samples_train, transform=self.train_transform)
        self.val_ds = USCutDatasetV2(self.dfs_valid, self.hparams.mount_point, img_column=self.hparams.img_column, num_samples=self.hparams.num_samples_val, transform=self.test_transform)
        self.test_ds = USCutDatasetV2(self.dfs_test, self.hparams.mount_point, img_column=self.hparams.img_column, num_samples=self.hparams.num_samples_test, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.hparams.drop_last, shuffle=True, prefetch_factor=self.hparams.prefetch_factor, collate_fn=self.collate_all)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, drop_last=self.hparams.drop_last, collate_fn=self.collate_all)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=1, num_workers=self.hparams.num_workers, collate_fn=self.collate_all)
    
class USButterflyBlindSweep(Dataset):
    def __init__(self, df, mount_point = "./", img_column='img_path', transform=None, num_frames=-1, continous_frames=False):
        self.df = df
        self.mount_point = mount_point
        self.transform = transform
        self.img_column = img_column
        self.keys = self.df.index
        self.num_frames = num_frames
        self.continous_frames = continous_frames 

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.mount_point, self.df.iloc[idx][self.img_column])

        try:
            img = sitk.ReadImage(img_path)
            img_np = sitk.GetArrayFromImage(img)
            img_t = torch.tensor(img_np, dtype=torch.float32)
            
            if img.GetNumberOfComponentsPerPixel() == 1:
                img_t = img_t.unsqueeze(-1)
            
            img_t = img_t.permute(3, 0, 1, 2)/255.0  # Change to (C, D, H, W)             
            
            if self.num_frames > 0:

                if self.continous_frames:
                    if img_t.shape[1] <= self.num_frames:
                        # pad with zeros if the number of frames is less than num_frames
                        pad_size = self.num_frames - img_t.shape[1]
                        img_t = torch.nn.functional.pad(img_t, (0, 0, 0, 0, 0, pad_size), mode='constant', value=0.0)
                        idx = torch.arange(0, self.num_frames)
                    else:
                        idx = torch.randint(low=0, high=img_t.shape[1] - self.num_frames, size=(1,)).item()
                        idx = torch.arange(idx, idx + self.num_frames)
                else:
                    idx = torch.randint(low=0, high=img_t.shape[1], size=self.num_frames)
                    idx = idx.sort().values

                img_t = img_t[:, idx, :, :]
                    
        except:
            print("Error reading cine: " + img_path)
            n = self.num_frames if self.num_frames > 0 else 1
            img_t = torch.zeros(1, n, 256, 256, dtype=torch.float32)

        if self.transform:
            img_t = self.transform(img_t)

        return img_t
    
class USButterflyBlindSweepDataModule(LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.df_train = pd.read_csv(self.hparams.csv_train)
        self.df_val = pd.read_csv(self.hparams.csv_valid)

        self.train_transform = None
        self.valid_transform = None

    @staticmethod
    def add_data_specific_args(parent_parser):

        group = parent_parser.add_argument_group("USButterflyBlindSweepDataModule")
        group.add_argument('--batch_size', type=int, default=2)
        group.add_argument('--num_workers', type=int, default=6)
        group.add_argument('--num_frames', type=int, default=64)
        group.add_argument('--continous_frames', type=int, default=1)
        group.add_argument('--img_column', type=str, default="img")
        group.add_argument('--csv_train', type=str, default=None, required=True)
        group.add_argument('--csv_valid', type=str, default=None, required=True)
        group.add_argument('--mount_point', type=str, default="./")
        group.add_argument('--drop_last', type=int, default=0)
        group.add_argument('--collate_batch', type=int, default=0)

        return parent_parser
        
    
    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        self.train_ds = USButterflyBlindSweep(self.df_train, self.hparams.mount_point, img_column=self.hparams.img_column, transform=self.train_transform, num_frames=self.hparams.num_frames, continous_frames=True)
        self.val_ds = USButterflyBlindSweep(self.df_val, self.hparams.mount_point, img_column=self.hparams.img_column, transform=self.valid_transform, num_frames=self.hparams.num_frames, continous_frames=True)
    
    def collate_batch(self, batch):
        return torch.cat(batch, dim=1).permute(1, 0, 2, 3) # Change to (B, C, H, W)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, persistent_workers=True, pin_memory=True, drop_last=bool(self.hparams.drop_last), shuffle=True, prefetch_factor=2, collate_fn=self.collate_batch if self.hparams.collate_batch else None)
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, drop_last=bool(self.hparams.drop_last), prefetch_factor=2, collate_fn=self.collate_batch if self.hparams.collate_batch else None)
    
class Cut3DDataModule(LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.df_train_diffusor = pd.read_csv(self.hparams.csv_train_diffusor)
        self.df_val_diffusor = pd.read_csv(self.hparams.csv_valid_diffusor)
        self.df_train = pd.read_csv(self.hparams.csv_train)
        self.df_val = pd.read_csv(self.hparams.csv_valid)

        self.train_transform = None
        self.valid_transform = None

    @staticmethod
    def add_data_specific_args(parent_parser):

        group = parent_parser.add_argument_group("Cut3DDataModule")
        group.add_argument('--batch_size', type=int, default=2)
        group.add_argument('--num_workers', type=int, default=6)
        group.add_argument('--num_frames', type=int, default=128)        
        group.add_argument('--img_column_diffusor', type=str, default="img")
        group.add_argument('--csv_train_diffusor', type=str, default=None, required=True)
        group.add_argument('--csv_valid_diffusor', type=str, default=None, required=True)
        group.add_argument('--img_column', type=str, default="file_path")
        group.add_argument('--csv_train', type=str, default=None, required=True)
        group.add_argument('--csv_valid', type=str, default=None, required=True)
        group.add_argument('--mount_point', type=str, default="./")
        group.add_argument('--drop_last', type=int, default=0)

        return parent_parser
    
    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders

        train_diff_ds = DiffusorSampleDataset(self.df_train_diffusor, self.hparams.mount_point, img_column=self.hparams.img_column_diffusor)
        val_diff_ds = DiffusorSampleDataset(self.df_val_diffusor, self.hparams.mount_point, img_column=self.hparams.img_column_diffusor)

        train_butterfly_ds = USButterflyBlindSweep(self.df_train, self.hparams.mount_point, img_column=self.hparams.img_column, num_frames=self.hparams.num_frames, continous_frames=True)
        val_butterfly_ds = USButterflyBlindSweep(self.df_val, self.hparams.mount_point, img_column=self.hparams.img_column, num_frames=self.hparams.num_frames, continous_frames=True)

        self.train_ds = ConcatDataset(train_diff_ds, train_butterfly_ds)
        self.val_ds = ConcatDataset(val_diff_ds, val_butterfly_ds)
    
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.hparams.drop_last, shuffle=True, prefetch_factor=2)
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, drop_last=self.hparams.drop_last, prefetch_factor=2)