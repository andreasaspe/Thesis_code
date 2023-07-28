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

    
def gaussian_kernel_3d(origins, meshgrid_dim, sigma=1):
    x,y,z = origins
    mesh_x,mesh_y,mesh_z = meshgrid_dim
    
    x_grid, y_grid, z_grid = np.meshgrid(np.arange(mesh_x), np.arange(mesh_y),np.arange(mesh_z),indexing='ij')
    kernel = np.exp(-((x_grid - x)**2 + (y_grid - y)**2 + (z_grid - z)**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel


dir_data = r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data' #'/zhome/bb/f/127616/Documents/Thesis/Preprocessed_data/'
Output_folder = r'C:\Users\PC\Documents\Andreas_s174197\heatmaps' #"/zhome/bb/f/127616/Documents/Thesis/Heatmaps/"

#Create output-folder if it does not exist
if not os.path.exists(Output_folder):
    os.makedirs(Output_folder)

all_subjects = []
for filename in listdir(dir_data):
    subject = filename.split("_")[0]
    all_subjects.append(subject)
all_subjects = np.unique(all_subjects)

heatmaps = []

for subject in all_subjects:
    print("       SUBJECT: "+str(subject)+"\n")
    # LOAD FILES
    # Define file names
    filename_img = [f for f in listdir(dir_data) if (f.startswith(subject) and f.endswith('img.nii.gz'))][0]
    filename_msk = [f for f in listdir(dir_data) if (f.startswith(subject) and f.endswith('msk.nii.gz'))][0]
    filename_ctd = [f for f in listdir(dir_data) if (f.startswith(subject) and f.endswith('json'))][0]
    # Load files
    img_nib = nib.load(os.path.join(dir_data,filename_img))
    msk_nib = nib.load(os.path.join(dir_data,filename_msk))
    ctd_list = load_centroids(os.path.join(os.path.join(dir_data,filename_ctd)))
    
    #Get data shape
    dim1, dim2, dim3 =  img_nib.header.get_data_shape()
    
    #Intialise heatmap
    heatmap = np.zeros((dim1,dim2,dim3))
    
    #Calculate heatmap
    for v in ctd_list[1:]:
        new_heatmap = gaussian_kernel_3d(origins = (v[1],v[2],v[3]), meshgrid_dim = (dim1,dim2,dim3), sigma = 2)
        new_heatmap = (new_heatmap - new_heatmap.min()) / (new_heatmap.max() - new_heatmap.min())
        heatmap += new_heatmap
        
    #Normalise heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    
    #Thresholding
    #heatmap[heatmap < 0.2] = 0
    
    heatmap_nifti = nib.Nifti1Image(heatmap, img_nib.affine)
    nib.save(heatmap_nifti, os.path.join(Output_folder, subject+'_heatmap.nii.gz'))
    
    # img_data = img_nib.get_fdata() #Load data
    
    # #Plot
    # for i in range(0,dim1,1):
    #     fig, ax = plt.subplots()
    #     ax.imshow(img_data[i,:,:].T,cmap="gray",origin="lower")
    #     ax.imshow(heatmap[i,:,:].T, cmap='hot', vmin=heatmap[i,:,:].T.min(), vmax=heatmap[i,:,:].T.max(),alpha = 1.0*(heatmap[i,:,:].T>0.4)) #Jo større tal ved alpha, jo bredere kurve
    #     ax.set_title('Dim1, '+str(subject)+", Slice: "+str(i))
    #     plt.show()
























    # counter = 0
    # for v in ctd_list[1:]:
    #     new_heatmap = gaussian_kernel_3d(origins = (v[1],v[2],v[3]), meshgrid_dim = (dim1,dim2,dim3), sigma = 15)
    #     new_heatmap = (new_heatmap - new_heatmap.min()) / (new_heatmap.max() - new_heatmap.min())
    #     heatmap += new_heatmap
    #     #heatmaps.append(new_heatmap)
        
    # heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    # img_data = img_nib.get_fdata() #Load data
    
    # #Plot
    # for i in range(217,255,1):
    #     fig, ax = plt.subplots()
    #     ax.imshow(img_data[i,:,:].T,cmap="gray",origin="lower")
    #     for j in range(len(heatmaps)):
    #         ax.imshow(heatmaps[j][i,:,:].T, cmap='hot', vmin=heatmaps[j][i,:,:].T.min(), vmax=heatmaps[j][i,:,:].T.max(),alpha = 1.0*(heatmaps[j][i,:,:].T>0.5))
    #     ax.set_title('Dim1, '+str(subject)+", Slice: "+str(i))
    #     plt.show()
        


            #ax.imshow(heatmaps[j][i,:,:].T, cmap='hot', vmin=heatmaps[j].min(), vmax=heatmaps[j].max(),alpha =0.5*(new_heatmap[i,:,:].T<0.2))
