#General imports
import os
import sys
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.transforms as transforms
#My own documents
from my_plotting_functions import *
from Create_dataset import *
from my_data_utils import Predict
from new_VertebraeLocalisationNet import *
#from My_networks.VertebraeLocalisation.Verse.VertebraeLocalisationNet import *

#For everything
# img_dir_validation = '/scratch/s174197/data/Verse20/VertebraeLocalisation/Verse20_validation_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_training_prep/img' #'/scratch/s174197/data/Verse20/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20_training_prep/img' #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data\img'
# heatmap_dir_validation = '/scratch/s174197/data/Verse20/VertebraeLocalisation/Verse20_validation_heatmaps'
img_dir_validation = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation/Verse20_validation_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_training_prep/img' #'/scratch/s174197/data/Verse20/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20_training_prep/img' #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data\img'
heatmap_dir_validation = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation/Verse20_validation_heatmaps'
# 
#Load data
# VerSe_train = LoadData(img_dir=img_dir_training, ctd_dir=ctd_dir_training, sigma=3)
# train_loader = DataLoader(VerSe_train, batch_size=1,
#                         shuffle=False, num_workers=0) #SET TO TRUE!

Verse_validation = LoadFullData(img_dir=img_dir_validation, heatmap_dir = heatmap_dir_validation,transform=None)
val_loader = DataLoader(Verse_validation, batch_size=1,
                        shuffle=False, num_workers=0) #SET TO TRUE!

#Define model
model = VertebraeLocalisationNet(0.0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #device = torch.device('cpu')
model.to(device)


#Define loss function and optimizer
loss_fn = nn.MSELoss()

#Batching through data
model.eval() #Set to evaluation
val_loss_batches = []
with torch.no_grad():
    for i, (img, heatmap, inputs_list, targets_list, start_end_voxels, subject) in enumerate(tqdm(val_loader)):
        _, dim1, dim2, dim3 = img.shape
        outputs_list = []
        for i in range(len(inputs_list)):
            #Initialise vector
            full_output_temp = torch.empty((1,8,dim1,dim2,dim3))
            #Unpack targets and inputs and get predictions
            inputs, targets = inputs_list[i].to(device), targets_list[i].to(device)
            output = model(inputs)
            #Get start and end voxel
            start_voxel = start_end_voxels[i][0].item()
            end_voxel = start_end_voxels[i][1].item()
            #Put into the output_tensor
            full_output_temp[:,:,:,:,start_voxel:end_voxel+1] = output
            #Append to list
            outputs_list.append(full_output_temp)
        #Taking the maximum response
        final_output = torch.max(*outputs_list)
            
            
            
            # #Handle overlap
            # if i == 0: #First time. OBS, måske crasher det, hvis den er præcis 128? Altså så vi ikke skal bruge mere end én patch.
            #     next_start_voxel = start_end_voxels[i+1][0]
            #     #Safe area
            #     full_output[:,:,:,:,start_voxel:next_start_voxel]
            #     #Save overlap area
            #     output_overlap_former = output[:,:,:,:,next_start_voxel:end_voxel+1]
            
            # elif i == len(inputs_list)-1: #Last time
            
            # else: #In the middle somewhere
            #     former_end_voxel = start_end_voxels[i-1][1]
            #     #Safe area
            #     full_output[:,:,:,:,former_end_voxel:next_start_voxel]
                
                
                
        
        
    

    # show_heatmap_img_dim1(inputs[0,0,:,:,:], targets[0,:,:,:], subject[0])
    # show_heatmap_img_dim2(inputs[0,0,:,:,:], targets[0,:,:,:], subject[0])
    # show_heatmap_img_dim3(inputs[0,0,:,:,:], targets[0,:,:,:], subject[0])
    print(subject[0])