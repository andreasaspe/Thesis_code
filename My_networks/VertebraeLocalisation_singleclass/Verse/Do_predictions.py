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
from functools import reduce 
#My own documents
from my_plotting_functions import *
#from new_VertebraeLocalisationNet import *
from Create_dataset import *
from my_data_utils import *
from new_VertebraeLocalisationNet_batchnormdropout import *
#from new_VertebraeLocalisationNet_batchnormdropout import *


#GPU-cluster
img_dir = '/scratch/s174197/data/Verse20/VertebraeLocalisation2/Verse20_test_prep/img' #'/scratch/s174197/data/Verse20/VertebraeLocalisation2/Verse20_training_prep2/img' #'/scratch/s174197/data/Verse20/VertebraeLocalisation2/Verse20_validation_prep_alldata/img' #'/scratch/s174197/data/Verse20/VertebraeLocalisation2/Verse20_validation_prep2/img' #'/scratch/s174197/data/Verse20/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20_training_prep/img' #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data\img'
heatmap_dir = '/scratch/s174197/data/Verse20/VertebraeLocalisation2/Verse20_test_heatmaps' #'/scratch/s174197/data/Verse20/VertebraeLocalisation2/Verse20_training_heatmaps2' #'/scratch/s174197/data/Verse20/VertebraeLocalisation2/Verse20_validation_heatmaps_alldata' #'/scratch/s174197/data/Verse20/VertebraeLocalisation2/Verse20_validation_heatmaps2'
output_pred_dir = '/scratch/s174197/data/Verse20/VertebraeLocalisation2/Verse20_test_predictions' #'/scratch/s174197/data/Verse20/VertebraeLocalisation2/for_making_figure/output' #'/scratch/s174197/data/Verse20/VertebraeLocalisation2/Verse20_validation_predictions_alldata' #'/scratch/s174197/data/Verse20/VertebraeLocalisation2/Verse20_validation_predictions2' #Prediction directory
#Two more for spatial and local, remember to change code later also, if you want this
# local_pred_dir = '/scratch/s174197/data/Verse20/VertebraeLocalisation2/for_making_figure/local' #'/scratch/s174197/data/Verse20/VertebraeLocalisation2/Verse20_validation_predictions_alldata' #'/scratch/s174197/data/Verse20/VertebraeLocalisation2/Verse20_validation_predictions2' #Prediction directory
# spatial_pred_dir = '/scratch/s174197/data/Verse20/VertebraeLocalisation2/for_making_figure/spatial' #'/scratch/s174197/data/Verse20/VertebraeLocalisation2/Verse20_validation_predictions_alldata' #'/scratch/s174197/data/Verse20/VertebraeLocalisation2/Verse20_validation_predictions2' #Prediction directory
#Checkpoint
checkpoint_parent_dir = '/scratch/s174197/data/Checkpoints/VertebraeLocalisation2/Only_elastic_earlystopping' #/VertebraeLocalisation2/alldata' #'/scratch/s174197/data/Checkpoints'
checkpoint_filename = 'Only_elastic_earlystopping_epoch1040_batchsize1_lr1e-05_wd0.0001.pth' #'Batchnorm_dropout_batchsize1_lr1e-05_wd0.0001.pth' #'Second_try_No_dropout_newinitialisation_batchsize1_lr1e-05_wd0.0001.pth' #'Third_try_No_dropout_newinitialisation142.pth' #'Second_try_No_dropout_newinitialisation_batchsize1_lr1e-05_wd0.0001.pth' #'NEWNetworkWithNewInitialisation_GPU_batchsize1_lr0.001_wd0.pth' #'FIXEDVAL_dropout3_higherdecay_batchsize1_lr1e-05_wd0.0001.pth' #'FIXEDVAL_dropout3_higherdecay_batchsize1_lr1e-05_wd0.0001.pth'

#mac
# img_dir = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation2/Verse20_validation_prep_alldata/img'
# heatmap_dir = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation2/Verse20_validation_heatmaps_alldata'
# output_pred_dir = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation2/Verse20_validation_predictions_alldata' #Prediction directory
# #Checkpoint
# checkpoint_parent_dir = '/Users/andreasaspe/Documents/Checkpoints/VertebraeLocalisation2/alldata' #'/scratch/s174197/data/Checkpoints'
# checkpoint_filename = 'Third_try_No_dropout_newinitialisation142' #'Second_try_No_dropout_newinitialisation_batchsize1_lr1e-05_wd0.0001.pth' #'NEWNetworkWithNewInitialisation_GPU_batchsize1_lr0.001_wd0.pth' #'FIXEDVAL_dropout3_higherdecay_batchsize1_lr1e-05_wd0.0001.pth' #'FIXEDVAL_dropout3_higherdecay_batchsize1_lr1e-05_wd0.0001.pth'


checkpoint_dir = os.path.join(checkpoint_parent_dir,checkpoint_filename)


#Load data
Data = LoadFullData(img_dir=img_dir,heatmap_dir = heatmap_dir)
loader = DataLoader(Data, batch_size=1,
                        shuffle=False, num_workers=0)
#Load model
model = VertebraeLocalisationNet(0.0)


    
if not os.path.exists(output_pred_dir):
            os.makedirs(output_pred_dir)
#For local and spatial
# if not os.path.exists(local_pred_dir):
#             os.makedirs(local_pred_dir)
# if not os.path.exists(spatial_pred_dir):
#             os.makedirs(spatial_pred_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #device = torch.device('cpu')
checkpoint = torch.load(checkpoint_dir,map_location=device)


#Send to GPU!
model.to(device)
# Load the saved weights into the model
model.load_state_dict(checkpoint['model_state_dict'])

# Set the model to evaluation mode
model.eval() 

with torch.no_grad():
    for i, (img, _, inputs_list, targets_list, start_end_voxels, subject) in enumerate(tqdm(loader)):
        assert len(subject) == 1 #Make sure we are only predicting one batch
        _, dim1, dim2, dim3 = img.shape
        outputs_list = []
        local_list = []
        spatial_list = []
        #Initialise vector
        full_output_temp = torch.ones((1,1,dim1,dim2,dim3))*(-1000)
        # full_local_temp = torch.ones((1,1,dim1,dim2,dim3))*(-1000)
        # full_spatial_temp = torch.ones((1,1,dim1,dim2,dim3))*(-1000)
        for j in range(len(inputs_list)):
            #Unpack targets and inputs and get predictions (I know I don't need the targets)
            inputs = inputs_list[j]
            targets = targets_list[j]
            #Sent to device
            inputs, targets = inputs.to(device), targets.to(device)
            #Forward pass
            output, _, _ = model(inputs) #output, local, spatial = model(inputs)

            #Get start and end voxel
            start_voxel = start_end_voxels[j][0].item()
            end_voxel = start_end_voxels[j][1].item()
            #Put into the output_tensor
            full_output_temp[:,:,:,:,start_voxel:end_voxel+1] = output
            # full_local_temp[:,:,:,:,start_voxel:end_voxel+1] = local
            # full_spatial_temp[:,:,:,:,start_voxel:end_voxel+1] = spatial
            #Append to list
            outputs_list.append(full_output_temp)
            # local_list.append(full_local_temp)
            # spatial_list.append(full_spatial_temp)
        #Taking the maximum response
        prediction = reduce(torch.max,outputs_list)
        # local_prediction = reduce(torch.max,local_list)
        # spatial_prediction = reduce(torch.max,spatial_list)
        #Save
        torch.save(prediction, os.path.join(output_pred_dir,subject[0] + '_heatmap_pred.pt'))
        # torch.save(local_prediction, os.path.join(local_pred_dir,subject[0] + '_heatmap_pred.pt'))
        # torch.save(spatial_prediction, os.path.join(spatial_pred_dir,subject[0] + '_heatmap_pred.pt'))










# IF YOU WANT IT IN A FUNCTION
# Predict
# Predict_VLN2(loader, model, checkpoint_dir, output_pred_dir)




# def Predict_VLN2(dataloader, model, checkpoint_dir, output_dir):
#     """
#     Predicts data for VerteBraeLocalisationNet2 and saves it

#     Arguments:
#     dataloader - a Pytorch dataloader with data to predict
#     model - the architecture of the network network defined as a pytorch model. For instance UNet3D()
#     checkpoint_dir - the directory for the file containing the checkpoint of the model
#     output_dir - the directory for where to save the output. Will create the folder if it does not exits.
#     """
    
#     if not os.path.exists(output_dir):
#                os.makedirs(output_dir)

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #device = torch.device('cpu')
#     checkpoint = torch.load(checkpoint_dir,map_location=device)

#     #Define model
#     model = model
#     #Send to GPU!
#     model.to(device)
#     # Load the saved weights into the model
#     model.load_state_dict(checkpoint['model_state_dict'])
    
    
#     # Set the model to evaluation mode
#     model.eval() 
    
#     with torch.no_grad():
#         for i, (img, _, inputs_list, targets_list, start_end_voxels, subject) in enumerate(tqdm(dataloader)):
#             assert len(subject) == 1 #Make sure we are only predicting one batch
#             _, dim1, dim2, dim3 = img.shape
#             outputs_list = []
#             for i in range(len(inputs_list)):
#                 #Initialise vector
#                 full_output_temp = torch.empty((1,8,dim1,dim2,dim3))
#                 #Unpack targets and inputs and get predictions (I know I don't need the targets)
#                 inputs = inputs_list[i]
#                 targets = targets_list[i]
#                 #Sent to devicexe
#                 inputs, targets = inputs.to(device), targets.to(device)
#                 #Forward pass
#                 output = model(inputs)
#                 #Get start and end voxel
#                 start_voxel = start_end_voxels[i][0].item()
#                 end_voxel = start_end_voxels[i][1].item()
#                 #Put into the output_tensor
#                 full_output_temp[:,:,:,:,start_voxel:end_voxel+1] = output
#                 #Append to list
#                 outputs_list.append(full_output_temp)
#             #Taking the maximum response
#             prediction = torch.max(*outputs_list)
#             #Save
#             torch.save(prediction, os.path.join(output_dir,subject[0] + '_heatmap_pred.pt'))