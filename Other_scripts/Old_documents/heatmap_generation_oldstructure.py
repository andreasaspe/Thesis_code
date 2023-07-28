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


    # """
    # Computes a 2D Gaussian kernel centered at the specified coordinates (x, y)
    # on a meshgrid with dimensions (meshgrid_dim, meshgrid_dim) and with a given standard deviation (sigma).
    
    # Args:
    #     x (float): x-coordinate of the center of the Gaussian kernel
    #     y (float): y-coordinate of the center of the Gaussian kernel
    #     meshgrid_dim (int): the dimensions of the meshgrid (output of np.meshgrid) on which to compute the kernel
    #     sigma (float): the standard deviation of the Gaussian kernel (default=1)
    
    # Returns:
    #     kernel (np.ndarray): 2D Gaussian kernel centered at (x, y) with dimensions (meshgrid_dim, meshgrid_dim)
    # """
    
    
def gaussian_kernel_3d(origins, meshgrid_dim, sigma=1):
    x,y,z = origins
    mesh_x,mesh_y,mesh_z = meshgrid_dim
    
    x_grid, y_grid, z_grid = np.meshgrid(np.arange(mesh_x), np.arange(mesh_y),np.arange(mesh_z),indexing='ij')
    kernel = np.exp(-((x_grid - x)**2 + (y_grid - y)**2 + (z_grid - z)**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel

# heatmap = gaussian_kernel_3d((10,20,30),(30,40,50),10)

# heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

# threshold = 0.5
# kernel[kernel < threshold] = 0
# kernel[kernel >= threshold] = 1


# for i in range(30):
#     fig, ax = plt.subplots()
#     ax.imshow(heatmap[i,:,:].T, cmap='hot', vmin=0, vmax=1)
#     plt.show()


dir_rawdata = '/Users/andreasaspe/Documents/Data/dataset-01training/rawdata'
dir_derivatives = '/Users/andreasaspe/Documents/Data/dataset-01training/derivatives'
Output_folder = "/Users/andreasaspe/Documents/Data/Preprocessed_data"

scans = [f for f in listdir(dir_rawdata) if f.startswith('sub')] 

heatmaps = []

for subject in scans:
    print("       SUBJECT: "+str(subject)+"\n")
    # LOAD FILES
    # Define file names
    filename_img = [f for f in listdir(os.path.join(dir_rawdata,subject)) if f.endswith('.gz')][0]
    filename_msk = [f for f in listdir(os.path.join(dir_derivatives,subject)) if f.endswith('.gz')][0]
    filename_ctd = [f for f in listdir(os.path.join(dir_derivatives,subject)) if f.endswith('.json')][0]
    
    # Load files
    img_nib = nib.load(os.path.join(dir_rawdata,subject,filename_img))
    msk_nib = nib.load(os.path.join(dir_derivatives,subject,filename_msk))
    ctd_list = load_centroids(os.path.join(os.path.join(dir_derivatives,subject,filename_ctd)))
    
    
    #Get data shape
    dim1, dim2, dim3 =  img_nib.header.get_data_shape()
    
    heatmap = np.zeros((dim1,dim2,dim3))
    
    for v in ctd_list[1:]:
        new_heatmap = gaussian_kernel_3d(origins = (v[1],v[2],v[3]), meshgrid_dim = (dim1,dim2,dim3), sigma = 15)
        new_heatmap = (new_heatmap - new_heatmap.min()) / (new_heatmap.max() - new_heatmap.min())
        heatmap += new_heatmap
        
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    
    #Set everything else to zero
    heatmap[heatmap < 0.4] = -

    img_data = img_nib.get_fdata() #Load data
    
    #Plot
    # for i in range(0,dim1,5):
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
