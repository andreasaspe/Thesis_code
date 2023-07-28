#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 20:06:50 2023

@author: andreasaspe
"""

import os
import nibabel as nib
import nibabel.orientations as nio
from data_utilities import *
# from nibabel.affines import apply_affine
# import numpy.linalg as npl
import pickle
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from my_plotting_functions import *
from tqdm import tqdm
from my_data_utils import *

#######################################################
#################### CONTROL PANEL ####################
#######################################################
#Define scans
all_scans = 1 #Set to 1 if you want to preprocess all scans

New_orientation = ('L', 'A', 'S')
#Define directories
#Cluster
data_type = 'test'
dir_data = '/scratch/s174197/data/Verse20/Verse20_'+data_type+'_unpacked' #'/scratch/s174197/data/Verse20/Verse20_training_unpacked' #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_test_unpacked' #'/scratch/s174197/data/Verse20/Verse20_test_unpacked' #'/scratch/s174197/data/Verse20/Verse20_test_unpacked' #'/Users/andreasaspe/Documents/Data/Verse20_training' #r'C:\Users\PC\Documents\Andreas_s174197\dataset-verse20training_unpacked' #'/zhome/bb/f/127616/Documents/Thesis/Rawdata_training'
Output_folder = '/scratch/s174197/data/Verse20/Verse20_'+data_type+'_unpacked_cropped'

#mac
# data_type = 'test'
# dir_data = '/Users/andreasaspe/Documents/Data/Verse20/Verse20_'+data_type+'_unpacked' #'/scratch/s174197/data/Verse20/Verse20_training_unpacked' #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_test_unpacked' #'/scratch/s174197/data/Verse20/Verse20_test_unpacked' #'/scratch/s174197/data/Verse20/Verse20_test_unpacked' #'/Users/andreasaspe/Documents/Data/Verse20_training' #r'C:\Users\PC\Documents\Andreas_s174197\dataset-verse20training_unpacked' #'/zhome/bb/f/127616/Documents/Thesis/Rawdata_training'
# Output_folder = '/scratch/s174197/data/Verse20/Verse20_'+data_type+'_unpacked_cropped'

#######################################################
#######################################################
#######################################################

#Define list of scans
if all_scans:
    all_subjects = []
    for filename in listdir(dir_data):
        subject = filename.split("_")[0]
        #if subject.find('verse') != -1:
        all_subjects.append(subject)
    all_subjects = np.unique(all_subjects)
    all_subjects = all_subjects[all_subjects != '.DS'] #Sorting out .DS
else:
    all_subjects = list_of_subjects




#Initialising list and dictionaries to save data on the way
dim1_list = []
dim2_list = []
dim3_list = []
restrictions_dict = {}

compatible_subjects = []
#FOR LOOP START
for subject in tqdm(all_subjects):
    print("\n\n")
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

    Old_orientation = nio.ornt2axcodes(nio.io_orientation(img_nib.affine)) #Image orientation

    #Reorient
    img_reoriented = reorient_to(img_nib, axcodes_to=New_orientation)
    msk_reoriented = reorient_to(msk_nib, axcodes_to=New_orientation)
    ctd_reoriented = reorient_centroids_to(ctd_list, img_reoriented)

    
    #Get data
    data_img = np.asanyarray(img_reoriented.dataobj, dtype=img_reoriented.dataobj.dtype)
    data_msk = np.asanyarray(msk_reoriented.dataobj, dtype=msk_reoriented.dataobj.dtype)
    
    zooms = img_reoriented.header.get_zooms()
    gap = 30
    offset_z = int(np.round(gap/zooms[2]))
    for ctd in ctd_list[1:]:
        if ctd[0] == 17:
            gap = 30
            z_coordinate = int(np.round(ctd[3]))
            data_img = data_img[:,:,:z_coordinate+offset_z]
            data_msk = data_img[:,:,:z_coordinate+offset_z]
            
    if not os.path.exists(Output_folder):
        os.makedirs(Output_folder)
        
    img_nib = nib.Nifti1Image(data_img,img_reoriented.affine)
    msk_nib = nib.Nifti1Image(data_msk,msk_reoriented.affine)

    #Reorient back
    # img_nib = reorient_to(img_nib, axcodes_to=Old_orientation)
    # msk_nib = reorient_to(msk_nib, axcodes_to=Old_orientation)
    
    nib.save(img_nib,os.path.join(Output_folder,subject+'_img.nii.gz'))
    nib.save(msk_nib,os.path.join(Output_folder,subject+'_msk.nii.gz'))
    save_centroids(ctd_reoriented,os.path.join(Output_folder,subject+'.json'))