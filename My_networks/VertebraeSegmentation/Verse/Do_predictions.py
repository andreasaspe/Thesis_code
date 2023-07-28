#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 00:42:50 2023

@author: andreasaspe
"""

#General imports
import os
import sys
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
import nibabel as nib
#My own documents
from my_plotting_functions import *
from VertebraeSegmentationNet import *
from Create_dataset import *
from my_data_utils import *

#GPU-cluster
data_dir = '/scratch/s174197/data/Verse20/VertebraeSegmentation/Verse20_test_prep'
output_pred_dir = '/scratch/s174197/data/Verse20/VertebraeSegmentation/Verse20_test_predictions' #Prediction directory
#Checkpoint
checkpoint_parent_dir = '/scratch/s174197/data/Checkpoints/VertebraeSegmentation' #'/scratch/s174197/data/Checkpoints/VertebraeSegmentation/First_try'
checkpoint_filename = 'Only_rotation_batchsize1_lr1e-05_wd0.0005.pth' #'First_try_step8450_batchsize1_lr1e-05_wd0.0005.pth' #Den her er den gamle gode: 'First_try_batchsize1_lr0.0001_wd0.0005.pth' #Husk .pth #'NEWNetworkWithNewInitialisation_GPU_batchsize1_lr0.001_wd0.pth' #'FIXEDVAL_dropout3_higherdecay_batchsize1_lr1e-05_wd0.0001.pth' #'FIXEDVAL_dropout3_higherdecay_batchsize1_lr1e-05_wd0.0001.pth'



#Defining subdirectories
img_dir = os.path.join(data_dir,'img')
heatmap_dir = os.path.join(data_dir,'heatmaps')
msk_dir = os.path.join(data_dir,'msk')


checkpoint_dir = os.path.join(checkpoint_parent_dir,checkpoint_filename)

#Load data
Data = LoadData(img_dir=img_dir, heatmap_dir=heatmap_dir, msk_dir = msk_dir,transform=None)
loader = DataLoader(Data, batch_size=1,
                        shuffle=False, num_workers=0)
#Load model
model = Unet3D(0.0).double()

#Predict
Predict_VSN(loader, model, checkpoint_dir, output_pred_dir)