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
from SpineLocalisationNet import *
from Create_dataset import LoadData_notarget
from my_data_utils import Predict

#GPU-cluster
img_dir = '/scratch/s174197/data/Verse20/SpineLocalisation/Verse20_training_prep/img' #'/scratch/s174197/data/Verse20/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20_training_prep/img' #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data\img'
checkpoint_dir = '/scratch/s174197/data/Checkpoints/SpineLocalisation/First_try2950_batchsize1_lr0.0001_wd0.0005.pth' #'/home/s174197/Thesis/My_code/My_networks/Spine_Localisation/Checkpoints/checkpoint_batchsize1_learningrate0.0001.pth' #'/Users/andreasaspe/Library/Mobile Documents/com~apple~CloudDocs/DTU/12.semester/Thesis/My_code/My_networks/Spine_Localisation/Checkpoints' #'/home/s174197/Thesis/My_code/My_networks/Spine_Localisation/Checkpoints' #'/Users/andreasaspe/Library/Mobile Documents/com~apple~CloudDocs/DTU/12.semester/Thesis/My_code/My_networks/Spine_Localisation/Checkpoints'
output_pred_dir = '/scratch/s174197/data/Verse20/SpineLocalisation/Verse20_training_predictions' #Prediction directory

#Load data
Data = LoadData_notarget(img_dir=img_dir)
loader = DataLoader(Data, batch_size=1,
                        shuffle=False, num_workers=0)
#Load model
model = Unet3D()

#Predict
Predict(loader, model, checkpoint_dir, output_pred_dir)