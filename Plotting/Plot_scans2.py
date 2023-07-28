#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 13:40:00 2023

@author: andreasaspe
"""

import os
import nibabel as nib
import nibabel.orientations as nio
from data_utilities import *
# from nibabel.affines import apply_affine
# import numpy.linalg as npl
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
from my_data_utils import *
from my_plotting_functions import *




#######################################################
#################### CONTROL PANEL ####################
#######################################################
#Define scans
all_scans = 0 #Set to 1 if you want to preprocess all scans
list_of_subjects = ['sub-verse768'] #List of subjects, 'sub-verse510' 500, verse605
#Dårlig opløsning!
#sub-verse510
#sub-verse544
#God opløsning:
#sub-verse807

#Define rescale and reorientation parameters
New_orientation = ('L', 'A', 'S')
New_voxel_size = 8 # [mm]

#Define preprocessing details and printing outputs
Print_info = 1 #Set to 1 to print a lot of info on each scan.

#Define directories
dir_img = '/Users/andreasaspe/Documents/Data/Verse20/Verse20_test_unpacked_cropped' #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_test_unpacked' #'/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation2/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_training_unpacked' #'/Users/andreasaspe/Documents/Data/Verse20/SpineLocalisation/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_training_unpacked' #'/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_training_unpacked' #'/Users/andreasaspe/Documents/Data/CTSpine1K/trainset/gt' #/Users/andreasaspe/Documents/Data/Verse20/Verse20_test_unpacked' #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data\img' #'/zhome/bb/f/127616/Documents/Thesis/Preprocessed_data'
# dir_msk = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation2/Verse20_training_prep/msk' #dir_img #'/Users/andreasaspe/Documents/Data/Verse20/SpineLocalisation/Verse20_training_prep/msk' #'/Users/andreasaspe/Documents/Data/CTSpine1K/trainset/gt' #dir_img #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data\msk'
# dir_ctd = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation2/Verse20_training_prep/ctd' #dir_img #'/Users/andreasaspe/Documents/Data/Verse20/SpineLocalisation/Verse20_training_prep/ctd' #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data\ctd'

#######################################################
#######################################################
#######################################################

#Define list of scans
if all_scans:
    all_subjects = []
    for filename in listdir(dir_img):
        subject = filename.split("_")[0]
        if subject.find('.DS') == -1 and subject.find('verse') != -1: #PLOTTER KUN VERSE. IKKE GL
            all_subjects.append(subject)
    all_subjects = np.unique(all_subjects)
else:
    all_subjects = list_of_subjects



#FOR LOOP START
for subject in all_subjects:
    print("       SUBJECT: "+str(subject)+"\n")
    # LOAD FILES
    filename_img = [f for f in listdir(dir_img) if (f.startswith(subject) and f.endswith('img.nii.gz'))][0] #and f.endswith('img.nii.gz'))
    img_nib = nib.load(os.path.join(dir_img,filename_img))
    # filename_msk = [f for f in listdir(dir_msk) if (f.startswith(subject) and f.endswith('msk.nii.gz'))][0] #and f.endswith('msk.nii.gz'))
    # msk_nib = nib.load(os.path.join(dir_msk,filename_msk))
    # filename_ctd = [f for f in listdir(dir_ctd) if (f.startswith(subject) and f.endswith('.json'))][0] #and f.endswith('.json'))
    # ctd_list = load_centroids(os.path.join(os.path.join(dir_ctd,filename_ctd)))
            
    #Get info
    zooms = img_nib.header.get_zooms() #Voxel sizes
    axs_code = nio.ornt2axcodes(nio.io_orientation(img_nib.affine)) #Image orientation
    data_shape_voxels = img_nib.header.get_data_shape() #Shape of data
    data_shape_mm = np.array(data_shape_voxels)*np.array(zooms) #Data measures
    data_type = img_nib.header.get_data_dtype() #Data type

        
    #Get data
    img_data = img_nib.get_fdata() #Load data
    # msk_data = msk_nib.get_fdata() #Load data
    
    show_slices_dim1(img_data,subject)
    
    # show_centroids_dim1(img_data, ctd_list, subject,no_slices=150)
    