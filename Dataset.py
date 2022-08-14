# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 14:28:33 2022

@author: Administrateur
"""

#%% dataset prepartion
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
