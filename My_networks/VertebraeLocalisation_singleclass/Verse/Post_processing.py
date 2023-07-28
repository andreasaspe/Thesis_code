#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 19:23:21 2023

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
from new_VertebraeLocalisationNet import *
from Create_dataset import LoadData
from my_data_utils import *
import importlib
import pickle
import cc3d


### ADJUST ###
heatmap_pred_dir = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation2/OLD/Verse20_validation_predictions_justforfun' #'/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation2/Verse20_validation_predictions2' #'/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation2/SUBSET_Verse20_validation_predictions_alldata'  #'/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation2/Working_code/Verse20_validation_predictions2' #Predictions or ground truth.
heatmap_target_dir = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation2/OLD/Verse20_validation_heatmaps2' #'/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation2/Working_code/Verse20_validation_heatmaps2''/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation2/SUBSET_Verse20_validation_heatmaps_alldata' #'/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation2/Working_code/Verse20_validation_heatmaps2' #Predictions or ground truth.
img_dir = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation2/OLD/Verse20_validation_prep2/img' #'/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation2/SUBSET_Verse20_validation_prep_alldata/img' #'/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation2/Working_code/Verse20_validation_prep2/img' #Image folder of prepped data

#Old good
# heatmap_pred_dir = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation2/Verse20_validation_predictions_justforfun'
# heatmap_target_dir = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation2/Verse20_validation_heatmaps2'
# img_dir = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation2/Verse20_validation_prep2/img'


all_scans = 1 #Set to 1 if you want to preprocess all scans
list_of_subjects = ['sub-verse618'] #618 580 539 522 511 529 - 522 var sgu svær! - verse505 gl279 524 går ud over! sub-verse570 sub-verse764 'sub-verse578' sub-verse563 og 60 List of subjects, 'sub-verse510'512 #verse526 var før og efter dårlig med bounding box
##############


#sub-gl090 - har den alle centroids??? Det har den sgu da ikke...
#SUB-VERSE824 VAR FORKERT FØR.. SE IGEN


#Define list of scans
if all_scans:
    all_subjects = []
    for filename in listdir(heatmap_target_dir): #img_dir
        subject = filename.split("_")[0]
        #if subject.find('verse') != -1: #PLOTTER KUN VERSE. IKKE GL
        all_subjects.append(subject)
    all_subjects = np.unique(all_subjects)
    all_subjects = all_subjects[all_subjects != '.DS']
else:
    all_subjects = list_of_subjects

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #device = torch.device('cpu')

for subject in tqdm(all_subjects):
    print(subject)
    print("\n\n\n")

    #Load predictions
    filename_heatmap = [f for f in listdir(heatmap_pred_dir) if f.startswith(subject)][0]
    heatmap_file_dir = os.path.join(heatmap_pred_dir, filename_heatmap)
    heatmap_data = torch.load(heatmap_file_dir, map_location=device)
    heatmap_data_prediction = heatmap_data.detach().numpy()
    #heatmap_data_prediction[heatmap_data_prediction > 1.3] = 0
    #Normalize
    #heatmap_data_prediction = (heatmap_data_prediction - heatmap_data_prediction.min()) / (heatmap_data_prediction.max() - heatmap_data_prediction.min())

    #Load target
    filename_heatmap = [f for f in listdir(heatmap_target_dir) if f.startswith(subject)][0]
    heatmap_file_dir = os.path.join(heatmap_target_dir, filename_heatmap)
    heatmap_nib = nib.load(heatmap_file_dir)
    heatmap_data_target = np.asanyarray(heatmap_nib.dataobj, dtype=np.float32)
    # #Normalize
    # heatmap_data = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())

    #Load image
    filename_img = subject + "_img.nii.gz"
    img_nib = nib.load(os.path.join(img_dir,filename_img))
    img_data = np.asanyarray(img_nib.dataobj, dtype=np.float32)
    
    heatmap_data_prediction = heatmap_data_prediction.squeeze()
    
    binary_img = np.where(heatmap_data_prediction > 0.3,1,0)
    
    #show_mask_dim1(binary_image,subject)
    #show_heatmap_img_dim1(img_data,heatmap_data_target[:,:,:],subject,no_slices=20,alpha=0.3)
    #show_heatmap_img_dim1(img_data,heatmap_data_prediction,subject,no_slices=20,alpha=0.3)
    #show_heatmap_dim1(heatmap_data_prediction,subject,no_slices=100,alpha=0.3)
    #show_slices_dim1(img_data,subject)
        
    blobs = cc3d.connected_components(binary_img,connectivity=6)

    no_unique= len(np.unique(blobs))
    
    min_z_coordinates=[]
    for label in range(1,no_unique):
        min_z_coordinates.append(np.min(np.where(blobs == label)[2]))
    #print(min_z_coordinates)
    
    #Lowest_blob_index
    Lowest_blob_index = np.argmin(min_z_coordinates)+1 #Fordi jeg sortede 0 fra før oppe i forløbet. Se mit range. 0 er baggrunds-class
    
    #Get mask of lowest blob
    mask_lowest_blob = np.where(blobs == Lowest_blob_index,1,0)
    
    #Get index of maximum value
    #heatmap_cropped = heatmap_data_prediction[mask_lowest_blob == 1]
    
    # Get indices of non-zero values in the binary mask
    nonzero_indices = np.nonzero(mask_lowest_blob)

    # Create a list of tuples containing the coordinates
    coordinates = list(zip(nonzero_indices[0], nonzero_indices[1], nonzero_indices[2]))

    # Sort the coordinates based on the corresponding values in heatmap_array
    sorted_coordinates = sorted(coordinates, key=lambda coord: heatmap_data_prediction[coord], reverse=True)

    # # Get the coordinates of the 10 highest values
    # top_10_coordinates = sorted_coordinates[:10]
    
    # avg_coordinate = np.array([
    #     sum(coord[0] for coord in top_10_coordinates) / len(top_10_coordinates),
    #     sum(coord[1] for coord in top_10_coordinates) / len(top_10_coordinates),
    #     sum(coord[2] for coord in top_10_coordinates) / len(top_10_coordinates)
    # ])
    
    
    # Get the coordinates of the highest value
    max_coordinate = sorted_coordinates[0]
    
    max_coordinate = np.array([
        max_coordinate[0],
        max_coordinate[1],
        max_coordinate[2]
    ])
    
    coordinates_list = [max_coordinate]
    
    #Check sequence
    x_coordinate = int(np.round(max_coordinate[0])) #Current x_coordinate
    y_coordinate = int(np.round(max_coordinate[1])) #Current y_coordinate
    z_coordinate = int(np.round(max_coordinate[2])) #Current z_coordinate
    
    _, _, dim3 = heatmap_data_prediction.shape
    while abs(z_coordinate+10 - dim3-1) > 6:
    
        
        x_range = [x_coordinate-20,x_coordinate+20]
        y_range = [y_coordinate-20,y_coordinate+20]
        z_range = [z_coordinate+10,z_coordinate+20]
        offset = np.array([x_range[0],y_range[0],z_range[0]])
        
        heatmap_cropped = heatmap_data_prediction[x_range[0]:x_range[1],y_range[0]:y_range[1],z_range[0]:z_range[1]]
        #show_heatmap_dim1(heatmap_cropped,subject)
        # Flatten the 3D array
        flattened_array = heatmap_cropped.flatten()
        
        #Check if it is worth looking for
        if max(flattened_array) < 0.1: #Det her er en anden, måske mere effektiv måde at skrive max på: flattened_array[top_10_indices[-1]]. Udnytter at jeg allerede har kørt argsort
            break  
        
        # Get the indices of the 10 highest values
        top_10_indices = np.argsort(flattened_array)[-5:]
        
        # Reshape the indices to their original 3D shape
        reshaped_indices = np.unravel_index(top_10_indices, heatmap_cropped.shape)
        
        # Get the corresponding heatmap values for getting the weights
        heatmap_values = flattened_array[top_10_indices]
        
        # Calculate the weighted average coordinate
        weighted_avg_coordinate = np.array([
            np.average(reshaped_indices[0], weights=heatmap_values),
            np.average(reshaped_indices[1], weights=heatmap_values),
            np.average(reshaped_indices[2], weights=heatmap_values)
        ])
        
        #Plus offset
        weighted_avg_coordinate = weighted_avg_coordinate + offset
        
        #Append to list
        coordinates_list.append(weighted_avg_coordinate)
        
        #Update coordinatess
        x_coordinate = int(np.round(weighted_avg_coordinate[0])) #Current x_coordinate
        y_coordinate = int(np.round(weighted_avg_coordinate[1])) #Current y_coordinate
        z_coordinate = int(np.round(weighted_avg_coordinate[2])) #Current z_coordinate
        
        # print(x_coordinate)
        # print(y_coordinate)
        # print(z_coordinate)
        
        # i+=1
        # print(i)
        

    show_centroids_new_dim1(img_data,(2,2,2),coordinates_list,subject)
    





    # Print the coordinates of the 10 highest values
    
    
    #heatmap_cropped = np.where(mask_lowest_blob == 1, heatmap_data_prediction, 0)
    
    #show_heatmap_dim1(heatmap_cropped,subject)
    
    
    
    
    ## STOP HERE AND PLOT IN DEBUG MODE TO SEE BLOBS
    # for i in range(blobs.shape[0]):
    #     fig, ax = plt.subplots()
    #     ax.imshow(blobs[i,:,:].T,cmap='jet',vmin=0, vmax=no_unique-1,origin="lower") #or cmap = 'viridis'
    #     plt.show()

    # for label in range(no_unique):
    #     area = np.count_nonzero(blobs == label)
    #     if area < 200:
    #         blobs[blobs==label] = 0
    #     # else:
    #     #     print(area)
    
    # for i in range(blobs.shape[0]):
    #     fig, ax = plt.subplots()
    #     ax.imshow(blobs[i,:,:].T,cmap='jet',vmin=0, vmax=no_unique-1,origin="lower") #or cmap = 'viridis'
    #     plt.show()
    # from scipy.ndimage import binary_closing, grey_erosion, binary_erosion
    
    # def remove_thin_lines_3d(volume, structure):
    #     # Define the kernel as a 1D line in all dimensions
    #     #kernel = np.ones((kernel_length, 1, 1), dtype=bool)
        
    #     # Perform 3D morphological erosion using the defined kernel
    #     removed_lines_volume = binary_erosion(volume,structure=structure)
    
    #     return removed_lines_volume

    # structure = np.ones((5, 5, 5), dtype=bool) #np.ones((1, 10, 1), dtype=bool)  # Example: 3x3x3 cube-shaped structure
        
    # # Perform 3D closing operation
    # eroded_volume = remove_thin_lines_3d(binary_image, structure)

    # show_mask_dim1(eroded_volume,subject)



#     #Load data
#     dataset = LoadData(img_dir=img_dir, heatmap_dir=heatmap_dir)
#     dataloader = DataLoader(dataset, batch_size=1,
#                             shuffle=True, num_workers=0) #SET TO True!

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #device = torch.device('cpu')
#     checkpoint = torch.load(checkpoint_dir,map_location=device)
    
#     #Define model
#     model = VertebraeLocalisationNet(0.0)
#     # Load the saved weights into the model
#     model.load_state_dict(checkpoint['model_state_dict'])
    
#     # Set the model to evaluation mode
#     model.eval() 
    
#     loss_fn = nn.MSELoss()
    
#     with torch.no_grad():
#         for inputs, targets, subject in tqdm(dataloader):
#             assert len(subject) == 1 #Make sure we are only predicting one batch
#             print()
#             print(subject[0])
#             predictions = model(inputs)
#             loss = loss_fn(predictions, targets)
#             predictions = predictions.squeeze()
#             predictions = predictions.detach().numpy()
            
#             targets = targets.squeeze()
#             inputs = inputs.squeeze(0,1)
            
#             # for i in range(8):
#             #     predictions[i,:,:,:] = (predictions[i,:,:,:] - predictions[i,:,:,:].min()) / (predictions[i,:,:,:].max() - predictions[i,:,:,:].min())
      
#             for i in range(8): #3,5
#                 if torch.max(targets[i,:,:,:])==1:
#                     #show_heatmap_img_dim1(inputs,targets[i,:,:,:],subject[0],no_slices=40)
#                     # show_heatmap_dim1(predictions[i,:,:,:],subject[0],alpha=0.0001,no_slices=40)
#                     show_heatmap_img_dim1(inputs,predictions[i,:,:,:],subject[0],no_slices=40)

             
#             print()
#             print(loss.item())
#             print()
            
#             # show_heatmap_dim1(targets[0,:,:,:],subject,no_slices=40)
#             # show_heatmap_dim1(predictions[0,:,:,:],subject,no_slices=40)

            
        
#     show_slices_dim1(img_data,subject,no_slices=40)

#         #show_heatmap_dim1(img_data,heatmap_data,subject,alpha=0.4,no_slices=40)
#         #show_heatmap_dim2(img_data,heatmap_data,subject,alpha=0.4,no_slices=40)
#         #show_heatmap_dim3(img_data,heatmap_data,subject,alpha=0.4,no_slices=60)

















# #from My_networks.VertebraeLocalisation.Verse.VertebraeLocalisationNet import *















    # while abs(z_coordinate - dim3) > 6:
        
    #     x_range = [x_coordinate-20,x_coordinate+20]
    #     y_range = [y_coordinate-20,y_coordinate+20]
    #     z_range = [z_coordinate+6,z_coordinate+25]
    #     offset = np.array([x_range[0],y_range[0],z_range[0]])
        
    #     heatmap_cropped = heatmap_data_prediction[x_range[0]:x_range[1],y_range[0]:y_range[1],z_range[0]:z_range[1]]
    #     # Flatten the 3D array
    #     flattened_array = heatmap_cropped.flatten()
        
    #     #Check if it is worth looking for
    #     if max(flattened_array) < 0.1: #Det her er en anden, måske mere effektiv måde at skrive max på: flattened_array[top_10_indices[-1]]. Udnytter at jeg allerede har kørt argsort
    #         break  
        
    #     # Get the indices of the 10 highest values
    #     top_10_indices = np.argsort(flattened_array)[-10:]
        
    #     # Reshape the indices to their original 3D shape
    #     reshaped_indices = np.unravel_index(top_10_indices, heatmap_cropped.shape)
        
    #     # Calculate the average coordinate
    #     avg_coordinate = np.array([
    #         np.mean(reshaped_indices[0]),
    #         np.mean(reshaped_indices[1]),
    #         np.mean(reshaped_indices[2])
    #     ])
        
    #     #Plus offset
    #     avg_coordinate = avg_coordinate + offset
        
    #     #Append to list
    #     coordinates_list.append(avg_coordinate)
        
    #     #Update coordinatess
    #     x_coordinate = int(np.round(avg_coordinate[0])) #Current x_coordinate
    #     y_coordinate = int(np.round(avg_coordinate[1])) #Current y_coordinate
    #     z_coordinate = int(np.round(avg_coordinate[2])) #Current z_coordinate
        
    #     # print(x_coordinate)
    #     # print(y_coordinate)
    #     # print(z_coordinate)
        
 