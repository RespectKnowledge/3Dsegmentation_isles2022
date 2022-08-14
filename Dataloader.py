# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 14:30:07 2022

@author: Abdul Qayyum
"""
########## dataloader 

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" 
from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    EnsureType,
    Invertd,
)
#from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob

set_determinism(seed=0)

### dataset
#data_dir='/content/drive/MyDrive/ISLES2022/dataset-ISLES22^public^unzipped^version'
#example_case=1
#dwi_path = os.path.join(data_dir, 'rawdata', 'sub-strokecase{}'.format("%04d" %example_case), 'ses-0001',
                #    'sub-strokecase{}_ses-0001_dwi.nii.gz'.format("%04d" % example_case))
#print(dwi_path)
# path_train_volumes = sorted(glob.glob(os.path.join(data_dir, "rawdata","*","*","*_dwi.nii.gz")))
# #path_train_volumes
# #path_train_segmentation = sorted(glob.glob(os.path.join(in_dir, "derivatives",'*',"*.nii.gz")))
# path_train_segmentation = sorted(glob.glob(os.path.join(data_dir, "derivatives","*","*","*.nii.gz")))

# train_files = [{"image": image_name, 'label': label_name} for image_name, label_name in zip(path_train_volumes, path_train_segmentation)]


import os
import pandas as pd

train_file=pd.read_csv('/home/imranr/monabdul/dataset/train_fold0_new.csv')
in_dir='/home/imranr/monabdul/dataset/dataset-ISLES22^public^unzipped^version'
path_train_volumes=[]
path_train_segmentation=[]
for i in range(0,len(train_file)):
  pathtrain=train_file['PatientID'][i]
  path_train_volumes.append(os.path.join(in_dir+pathtrain))
  #pathtrain.replace('image','label')
  p1=pathtrain.replace('rawdata','derivatives')[:41]
  p2=pathtrain.replace('rawdata','derivatives').split('/')[-1].split('_')[0]+'_ses-0001_msk.nii.gz'
  paths=p1+p2
  #print(paths)
  path_train_segmentation.append(os.path.join(in_dir+paths))
  #break
import os
import pandas as pd
import pandas as pd
valid_file=pd.read_csv('/home/imranr/monabdul/dataset/valid_fold0_new.csv')
#in_dir='/content/drive/MyDrive/ISLES2022/dataset-ISLES22^public^unzipped^version'
path_valid_volumes=[]
path_valid_segmentation=[]
for i in range(0,len(valid_file)):
  pathvalid=valid_file['PatientID'][i]
  path_valid_volumes.append(os.path.join(in_dir+pathvalid))
  #pathtrain.replace('image','label')
  p1=pathvalid.replace('rawdata','derivatives')[:41]
  p2=pathvalid.replace('rawdata','derivatives').split('/')[-1].split('_')[0]+'_ses-0001_msk.nii.gz'
  paths=p1+p2
  #print(paths)
  path_valid_segmentation.append(os.path.join(in_dir+paths))
  #break
  
train_files = [{"image": image_name, 'label': label_name} for image_name, label_name in zip(path_train_volumes, path_train_segmentation)]

valid_files = [{"image": image_name, 'label': label_name} for image_name, label_name in zip(path_valid_volumes, path_valid_segmentation)]

import numpy as np
from monai.transforms import (
    Compose,
    LoadImaged,
    AddChanneld,
    CropForegroundd,
    Spacingd,
    Orientationd,
    SpatialPadd,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    RandRotated,
    RandZoomd,
    CastToTyped,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    RandFlipd,
    ToTensord,
    ScaleIntensityRanged,
    Resized,RandShiftIntensityd,
)
from monai.utils import first
import numpy as np
import nibabel as nib
from monai.transforms import(
    Compose,
    AddChanneld,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
)
from monai.data import DataLoader, Dataset, CacheDataset
from monai.utils import set_determinism
set_determinism(seed=0)
import os
import nibabel as nib
import glob
#image_keys: ["image"]
#all_keys: ["image", "label"]

generat_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        #AddChanneld(keys=["image", "label"]),
        AddChanneld(keys=["image","label"]),
        Spacingd(keys=["image", "label"], pixdim=(2.0, 2.0, 2.0), mode=("bilinear", "nearest")),
        #Spacingd(keys=["image", "label"],pixdim=(0.68825, 0.68825, 2.0),mode=("bilinear",) * len(['image']) + 
        #         ("nearest",),),
        #Orientationd(keys=["image", "label"], axcodes="RAS"),
        CropForegroundd(keys=['image','label'], source_key='image'),
        SpatialPadd(keys=["image", "label"], spatial_size=[96,96,96]),
        #Resized(keys=["image", "label"], spatial_size=[112,112,32]),
        ScaleIntensityRanged(keys=["image"], a_min=68.42, a_max=897.0,b_min=0.0, b_max=1.0, clip=True,),
        RandCropByPosNegLabeld(  # crop with center in label>0 with proba pos / (neg + pos)
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96,96,96),
            pos=1,
            neg=0,  # never center in background voxels
            num_samples=4,
            image_key=None,  # for no restriction with image thresholding
            image_threshold=0,
        ), 
        RandGaussianNoised(keys=["image"], mean=0., std=0.1, prob=0.2),
        RandGaussianSmoothd(
            keys=["image"],
            sigma_x=(0.5, 1.15),
            sigma_y=(0.5, 1.15),
            sigma_z=(0.5, 1.15),
            prob=0.2,
          ),
          RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
            ),
        RandAdjustContrastd(  # same as Gamma in nnU-Net
            keys=["image"],
            gamma=(0.7, 1.5),
            prob=0.3,
          ),
        RandZoomd(
            keys=["image", "label"],
            min_zoom=0.7,
            max_zoom=1.5,
            mode=("trilinear",) * len(["image"]) + ("nearest",),
            align_corners=(True,) * len(["image"]) + (None,),
            prob=0.3,
          ),
          #RandRotated(
            #keys=["image", "label"],
            #range_x=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            #range_y=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            #range_z=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            #mode=("bilinear",) * len(["image"]) + ("nearest",),
            #align_corners=(True,) * len(["image"]) + (None,),
            #padding_mode=("border", ) * len(["image", "label"]),
            #prob=0.3,
            #),
        CastToTyped(keys=["image", "label"], dtype=(np.float32,) * len(["image"]) + (np.uint8,)),
        #RandFlipd(keys=["image"], spatial_axis=[0, 1, 2], prob=0.5),
        #RandAffined(keys=['image', 'label'], prob=0.5, translate_range=10), 
        #RandRotated(keys=['image', 'label'], prob=0.5, range_x=10.0),
        #RandGaussianNoised(keys='image', prob=0.5),
        #NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ToTensord(keys=["image", "label"]),
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        #AddChanneld(keys=["image", "label"]),
        AddChanneld(keys=["image","label"]),
        #Resized(keys=["image", "label"], spatial_size=[128,128,128]),
        #Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(2.0, 2.0, 2.0),
            mode=("bilinear", "nearest"),
        ),
        #Resized(keys=["image", "label"], spatial_size=[128,128,128]),
        ScaleIntensityRanged(
            keys=["image"], a_min=68.42, a_max=897.0, b_min=0.0, b_max=1.0, clip=True
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        ToTensord(keys=["image", "label"]),
    ]
)
my_transform_org=Compose([LoadImaged(keys=["image", "label"]),AddChanneld(keys=["image", "label"]),ToTensord(keys=["image", "label"])])
train_ds = Dataset(data=train_files, transform=generat_transforms)
train_loader = DataLoader(train_ds, batch_size=1)
generat_patient = first(train_loader)

original_ds = Dataset(data=valid_files, transform=val_transforms)
val_loader = DataLoader(original_ds, batch_size=1)