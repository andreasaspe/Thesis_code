#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 11:16:50 2023

@author: andreasaspe
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from os import listdir
from data_utilities import *
from functools import reduce


def gaussian_kernel_3d(origins, meshgrid_dim, sigma=1):
    x,y,z = origins
    mesh_x,mesh_y,mesh_z = meshgrid_dim
    
    x_grid, y_grid, z_grid = np.meshgrid(np.arange(mesh_x), np.arange(mesh_y),np.arange(mesh_z),indexing='ij')
    kernel = np.exp(-((x_grid - x)**2 + (y_grid - y)**2 + (z_grid - z)**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel


#Dir_data er hovedmappen. Så definerer jeg img, msk og ctd længere nede selv.
dir_data = '/scratch/s174197/data/Verse20/VertebraeLocalisation/Verse20_validation_prep_new' #"/Users/andreasaspe/Documents/Data/Verse20/Verse20_training_prep" #"/Users/andreasaspe/Documents/Data/Verse20_training_prep" #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data' #PARENT FOLDER #'/zhome/bb/f/127616/Documents/Thesis/Preprocessed_data/'
Output_folder = '/scratch/s174197/data/Verse20/VertebraeLocalisation/Verse20_validation_heatmaps_new' #"/Users/andreasaspe/Documents/Data/Verse20_training_heatmaps" #r'C:\Users\PC\Documents\Andreas_s174197\heatmaps' #"/zhome/bb/f/127616/Documents/Thesis/Heatmaps/"

#Create output-folder if it does not exist
if not os.path.exists(Output_folder):
    os.makedirs(Output_folder)
heatmaps = []

img_dir = os.path.join(dir_data,'img')
msk_dir = os.path.join(dir_data,'msk')
ctd_dir = os.path.join(dir_data,'ctd')

for filename in os.listdir(img_dir):
    subject = filename.split("_")[0]
    print("       SUBJECT: "+str(subject)+"\n")
    # LOAD FILES

    # Define file names
    filename_img = filename
    #filename_msk = subject + '_msk.nii.gz'
    filename_ctd = subject + '_ctd.json'

    # Load files
    img_nib = nib.load(os.path.join(img_dir,filename_img))
    #msk_nib = nib.load(os.path.join(msk_dir,filename_msk))
    ctd_list = load_centroids(os.path.join(ctd_dir,filename_ctd))
    
    #Get data shape
    dim1, dim2, dim3 =  img_nib.header.get_data_shape()

    #Initialise target
    no_targets = 8
    targets = np.zeros((no_targets, dim1, dim2, dim3))
    #Loop through centroids
    for ctd in ctd_list[1:]:
        if 17 <= ctd[0] <= 24:
            ctd_index = ctd[0]-17
            heatmap = gaussian_kernel_3d(origins = (ctd[1],ctd[2],ctd[3]), meshgrid_dim = (dim1,dim2,dim3), sigma = 5)
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            targets[ctd_index ,:,:,:] = heatmap
    
    # #Thresholding
    # #heatmap[heatmap < 0.2] = 0
    
    targets_nifti = nib.Nifti1Image(targets, img_nib.affine)
    nib.save(targets_nifti, os.path.join(Output_folder, subject+'_heatmap.nii.gz'))
    




    #FOR PLOTTING
    # img_data = img_nib.get_fdata() #Load data
    
    # #Plot
    # for i in range(0,dim1,1):
    #     fig, ax = plt.subplots()
    #     max_val = img_data.max()
    #     min_val = img_data.min()
    #     ax.imshow(img_data[i,:,:].T,cmap="gray",origin="lower",vmin=min_val,vmax=max_val)
    #     ax.imshow(heatmap[i,:,:].T, cmap='hot', vmin=heatmap.T.min(), vmax=heatmap.T.max(),alpha = 1.0*(heatmap[i,:,:].T>0.4)) #Jo større tal ved alpha, jo bredere kurve
    #     ax.set_title('Dim1, '+str(subject)+", Slice: "+str(i))
    #     plt.show()