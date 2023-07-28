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
from my_data_utils import *
from functools import reduce



#Dir_data er hovedmappen. Så definerer jeg img, msk og ctd længere nede selv.
#gpu cluster
dir_data = '/scratch/s174197/data/Verse20/VertebraeLocalisation2/Verse20_test_prep' #"/Users/andreasaspe/Documents/Data/Verse20/Verse20_training_prep" #"/Users/andreasaspe/Documents/Data/Verse20_training_prep" #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data' #PARENT FOLDER #'/zhome/bb/f/127616/Documents/Thesis/Preprocessed_data/'
Output_folder = '/scratch/s174197/data/Verse20/VertebraeLocalisation2/Verse20_test_heatmaps'
sigma = 3

#mac
# dir_data = "/Users/andreasaspe/Documents/Data/Verse20/Data_for_figures"
# Output_folder = "/Users/andreasaspe/Documents/Data/Verse20/Data_for_figures/output_heatmap" #"/Users/andreasaspe/Documents/Data/Verse20_training_heatmaps" #r'C:\Users\PC\Documents\Andreas_s174197\heatmaps' #"/zhome/bb/f/127616/Documents/Thesis/Heatmaps/"

#Create output-folder if it does not exist
if not os.path.exists(Output_folder):
    os.makedirs(Output_folder)

img_dir = os.path.join(dir_data,'img')
ctd_dir = os.path.join(dir_data,'ctd')

for filename in os.listdir(img_dir):
    subject = filename.split("_")[0]
    print("       SUBJECT: "+str(subject)+"\n")

    # LOAD FILES
    # Define file names
    filename_img = filename
    filename_ctd = subject + '_ctd.json'

    # Load files
    img_nib = nib.load(os.path.join(img_dir,filename_img))
    ctd_list = load_centroids(os.path.join(ctd_dir,filename_ctd))
    
    #Get data shape
    dim1, dim2, dim3 =  img_nib.header.get_data_shape()

    # heatmap = np.zeros((dim1, dim2, dim3)) #UNCOMMENT IF YOU WANT SUM
    heatmaps_list = [] #UNCOMMENT IF YOU WANT MAXIMUM
    #Loop through centroids
    for ctd in ctd_list[1:]:
        if 17 <= ctd[0] <= 24:
            new_heatmap = gaussian_kernel_3d_new(origins = (ctd[1],ctd[2],ctd[3]), meshgrid_dim = (dim1,dim2,dim3), gamma=1, sigma = sigma)
            new_heatmap = (new_heatmap - new_heatmap.min()) / (new_heatmap.max() - new_heatmap.min())
            # heatmap += new_heatmap #UNCOMMENT IF YOU WANT SUM
            heatmaps_list.append(new_heatmap) #UNCOMMENT IF YOU WANT MAXMIMUM
    
    heatmap = reduce(np.maximum,heatmaps_list) #UNCOMMENT IF YOU WANT MAXIMUM *heatmaps_list
    
    #Thresholding
    heatmap[heatmap < 0.001] = 0

    heatmap_nifti = nib.Nifti1Image(heatmap, img_nib.affine)
    nib.save(heatmap_nifti, os.path.join(Output_folder, subject+'_heatmap.nii.gz'))
     