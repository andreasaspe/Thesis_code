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
import math
from data_utilities import *
from functools import reduce


def gaussian_kernel_3d_new(origins, meshgrid_dim, gamma = 1, sigma=1):
    d=3 #dimension
    x,y,z = origins
    mesh_x,mesh_y,mesh_z = meshgrid_dim
    
    x_grid, y_grid, z_grid = np.meshgrid(np.arange(mesh_x), np.arange(mesh_y),np.arange(mesh_z),indexing='ij')
    kernel = np.exp(-((x_grid - x)**2 + (y_grid - y)**2 + (z_grid - z)**2) / (2 * sigma**2))
    factor = gamma/( (2*math.pi)**(d/2)*sigma**d   )
    heatmap = factor*kernel
    return heatmap

#Dir_data er hovedmappen. Så definerer jeg img, msk og ctd længere nede selv.
dir_data = '/scratch/s174197/data/Verse20/SpineLocalisation/Verse20_test_prep' #"/Users/andreasaspe/Documents/Data/Verse20/Verse20_training_prep" #"/Users/andreasaspe/Documents/Data/Verse20_training_prep" #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data' #PARENT FOLDER #'/zhome/bb/f/127616/Documents/Thesis/Preprocessed_data/'
Output_folder = '/scratch/s174197/data/Verse20/SpineLocalisation/Verse20_test_heatmaps' #"/Users/andreasaspe/Documents/Data/Verse20_training_heatmaps" #r'C:\Users\PC\Documents\Andreas_s174197\heatmaps' #"/zhome/bb/f/127616/Documents/Thesis/Heatmaps/"

#Create output-folder if it does not exist
if not os.path.exists(Output_folder):
    os.makedirs(Output_folder)
heatmaps = []

img_dir = os.path.join(dir_data,'img')
ctd_dir = os.path.join(dir_data,'ctd')

for filename in os.listdir(img_dir):
    subject = filename.split("_")[0]
    print("       SUBJECT: "+str(subject)+"\n")

    # Define file names
    filename_img = filename
    filename_ctd = subject + '_ctd.json'

    # Load files
    img_nib = nib.load(os.path.join(img_dir,filename_img))
    ctd_list = load_centroids(os.path.join(ctd_dir,filename_ctd))
    
    #Get data shape
    dim1, dim2, dim3 =  img_nib.header.get_data_shape()
    
    #Intialise heatmap
    #heatmaps_list = [] #UNCOMMENT IF YOU WANT MAXIMUM
    heatmap = np.zeros((dim1,dim2,dim3)) #UNCOMMENT IF YOU WANT SUM
    
    #Calculate heatmap
    for v in ctd_list[1:]:
        new_heatmap = gaussian_kernel_3d_new(origins = (v[1],v[2],v[3]), meshgrid_dim = (dim1,dim2,dim3), gamma=1, sigma = 3)
        new_heatmap = (new_heatmap - new_heatmap.min()) / (new_heatmap.max() - new_heatmap.min())
        heatmap += new_heatmap #UNCOMMENT IF YOU WANT SUM
        #heatmaps_list.append(new_heatmap) #UNCOMMENT IF YOU WANT MAXMIMUM
    
    #heatmap = reduce(np.maximum,heatmaps_list) #UNCOMMENT IF YOU WANT MAXIMUM *heatmaps_list
                
    #Normalise heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    
    #Thresholding
    heatmap[heatmap < 0.001] = 0
    
    heatmap_nifti = nib.Nifti1Image(heatmap, img_nib.affine)
    nib.save(heatmap_nifti, os.path.join(Output_folder, subject+'_heatmap.nii.gz'))
    
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