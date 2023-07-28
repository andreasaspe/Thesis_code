#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 13:40:16 2023

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
import torch
import pickle
from tqdm import tqdm
#My functions
from my_data_utils import BoundingBox, RescaleBoundingBox
from my_plotting_functions import *




#######################################################
#################### CONTROL PANEL ####################
#######################################################
#Define scans
all_scans = 1 #Set to 1 if you want to preprocess all scans
list_of_subjects = ['sub-verse820'] #List of subjects 521


#Define directories


### GPU CLUSTER ###
dir_heatmap = '/scratch/s174197/data/Verse20/SpineLocalisation/Verse20_training_heatmaps' #'/Users/andreasaspe/Documents/Data/Verse20/SpineLocalisation/Verse20_test_predictions' #'/scratch/s174197/data/Verse20/Verse20_test_predictions'  #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_test_predictions' #'/scratch/s174197/data/Verse20/Verse20_test_unpacked' #'/scratch/s174197/data/Verse20/Verse20_test_unpacked' #'/Users/andreasaspe/Documents/Data/Verse20_training' #r'C:\Users\PC\Documents\Andreas_s174197\dataset-verse20training_unpacked' #'/zhome/bb/f/127616/Documents/Thesis/Rawdata_training'
dir_padding_specifications = '/scratch/s174197/data/Verse20/SpineLocalisation/Padding_specifications/pad_training'
dir_data = '/scratch/s174197/data/Verse20/Verse20_training_unpacked' #'/scratch/s174197/data/Verse20/Verse20_test_unpacked' #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_test_unpacked'
#Output folders
Output_folder = '/scratch/s174197/data/Verse20/VertebraeLocalisation/Verse20_validation_prep_new' #'/scratch/s174197/data/Verse20/Verse20_test_prep' #'/Users/andreasaspe/Documents/Data/Verse20_training_prep' #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data' #"/Users/andreasaspe/Documents/Data/Preprocessed_data"
Padding_output_directory = '/scratch/s174197/data/Verse20/VertebraeLocalisation/Padding_specifications'
Padding_output_filename = 'pad_validation'

### MAC ###
# dir_heatmap = '/Users/andreasaspe/Documents/Data/Verse20/SpineLocalisation/Verse20_training_heatmaps' #'/scratch/s174197/data/Verse20/Verse20_test_predictions'  #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_test_predictions' #'/scratch/s174197/data/Verse20/Verse20_test_unpacked' #'/scratch/s174197/data/Verse20/Verse20_test_unpacked' #'/Users/andreasaspe/Documents/Data/Verse20_training' #r'C:\Users\PC\Documents\Andreas_s174197\dataset-verse20training_unpacked' #'/zhome/bb/f/127616/Documents/Thesis/Rawdata_training'
# dir_data = '/Users/andreasaspe/Documents/Data/Verse20/Verse20_training_unpacked'
# Output_folder = '/Users/andreasaspe/Documents/Data/VertebraeLocalisation/ONLYONESAMPLE_Verse20_training_prep' #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data' #"/Users/andreasaspe/Documents/Data/Preprocessed_data"
# dir_padding_specifications = '/Users/andreasaspe/Documents/Data/Verse20/SpineLocalisation/Padding_specifications/pad_training'


#Define rescale and reorientation parameters
New_orientation = ('L', 'A', 'S')
New_voxel_size = 2 # [mm]

#Preprocessing
normalize_HU = 1 #Set to 1 if you want to normalize to the below range
HU_range_normalize = [-1, 1]
HU_cutoff = 1 #Set to 1 if you want to cut_off_HU. Define below:
HU_range_cutoff = [-200, 1000] #Define HU range. Only takes effekt if HU_cutoff is set to 1
pad_value = -1024 #'minimum' #Put in number or the string 'minimum' for padding with the minimum value in volume
dim1_new = 96
dim2_new = 96
dim3_new = 128
#######################################################
#######################################################
#######################################################


plt.style.use('default')


#Define list of scans
if all_scans:
    all_subjects = []
    for filename in listdir(dir_data):
        subject = filename.split("_")[0]
        #if subject.find('verse') != -1: #PLOTTER KUN VERSE. IKKE GL
        all_subjects.append(subject)
    all_subjects = np.unique(all_subjects)
    #Sorterer fil '.DS' fra
    all_subjects = all_subjects[all_subjects != '.DS']
else:
    all_subjects = list_of_subjects
    

#Load padding_specifications
with open(dir_padding_specifications, 'rb') as f:
    padding_specifications = pickle.load(f) 

restrictions = {}

for subject in tqdm(all_subjects):

    ctd_overview = []

    print("\n\n")
    print("       SUBJECT: "+str(subject)+"\n")

    #LOAD CENTROIDS
    filename_ctd = [f for f in listdir(dir_data) if (f.startswith(subject) and f.endswith('json'))][0]
    ctd_list = load_centroids(os.path.join(os.path.join(dir_data,filename_ctd)))

    try: #Prøver. Hvis ikke filen findes, så er det fordi jeg frasortede den på et tidspunkt.
        filename_heatmap = [f for f in listdir(dir_heatmap) if f.startswith(subject)][0]
        heatmap_file_dir = os.path.join(dir_heatmap, filename_heatmap)
    except:
        continue

    #LOAD HEATMAP
    try: #Load tensor or numpy
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #device = torch.device('cpu')
        heatmap_data = torch.load(heatmap_file_dir, map_location=device)
        heatmap_data = heatmap_data.detach().numpy()
        #Normalize
        heatmap_data = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())
    except:
        heatmap_nib = nib.load(heatmap_file_dir)
        heatmap_data = np.asanyarray(heatmap_nib.dataobj, dtype=np.float32)
        #Normalize
        heatmap_data = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())

    #LOAD IMAGE
    filename_img = [f for f in listdir(dir_data) if (f.startswith(subject) and f.endswith('img.nii.gz'))][0]
    img_nib = nib.load(os.path.join(dir_data,filename_img))
    # img_data = np.asanyarray(img_nib.dataobj, dtype=img_nib.dataobj.dtype) #Load 
    

    #FIND BOUNDING BOX
    old_restrictions = padding_specifications[subject]
    old_zooms = (8,8,8)
    new_zooms = img_nib.header.get_zooms()
    bb_coordinates, COM = BoundingBox(heatmap_data,old_restrictions)
    original_bb_coordinates, original_COM = RescaleBoundingBox(new_zooms,old_zooms,bb_coordinates,COM,old_restrictions)

    #show_boundingbox_dim1(img_data, original_bb_coordinates, subject)
    
    new_zooms = (2,2,2)
    old_zooms = img_nib.header.get_zooms()
    new_bb_coordinates, new_COM = RescaleBoundingBox(new_zooms,old_zooms,original_bb_coordinates,original_COM)


    #Guassian smoothing
    # img_data = gaussian_filter(img_data, sigma=[0.75/new_zooms[0] , 0.75/new_zooms[1], 0.75/new_zooms[2]]) #3/8 er bedre
    # img_nib = nib.Nifti1Image(img_data, img_nib.affine)


    #RESAMPLE AND REORIENT
    vs = (New_voxel_size,New_voxel_size,New_voxel_size)
    #Image
    img_resampled = resample_nib(img_nib, voxel_spacing=vs, order=3)
    img_resampled_reoriented = reorient_to(img_resampled, axcodes_to=New_orientation)
    #Centroids
    ctd_resampled = rescale_centroids(ctd_list, img_nib, vs)
    ctd_resampled_reoriented = reorient_centroids_to(ctd_resampled, img_resampled_reoriented)

    #Load data
    data_img = np.asanyarray(img_resampled_reoriented.dataobj, dtype=img_resampled_reoriented.dataobj.dtype)

    #Load and crop data
    #FROM COM
    # new_x_COM, new_y_COM, new_z_COM = new_COM
    # new_x_COM = np.round(new_x_COM).astype(int)
    # new_y_COM = np.round(new_y_COM).astype(int)
    # new_z_COM = np.round(new_z_COM).astype(int)
    # dim1, dim2, dim3 = data_img.shape
    # x_range = [max(new_x_COM-48,0),min(new_x_COM+48,dim1)]
    # y_range = [max(new_y_COM-48,0),min(new_y_COM+48,dim2)]
    # z_range = [0,0]
    # data_img = data_img[x_range[0]:x_range[1],y_range[0]:y_range[1],:]
    #FROM BOUNDING BOX
    x_min, x_max, y_min, y_max, z_min, z_max = new_bb_coordinates
    x_min = np.round(x_min).astype(int)
    x_max = np.round(x_max).astype(int)
    y_min = np.round(y_min).astype(int)
    y_max = np.round(y_max).astype(int)
    z_min = np.round(z_min).astype(int)
    z_max = np.round(z_max).astype(int)
    x_range = [x_min,x_max]
    y_range = [y_min,y_max]
    z_range = [z_min,z_max]
    data_img = data_img[x_range[0]:x_range[1],y_range[0]:y_range[1],z_range[0]:z_range[1]]
    
    #Get dimensions after cropping
    dim1, dim2, dim3 = data_img.shape

    #Calculate padding in each side (volume should be centered)
    padding_dim1 = (dim1_new-dim1)/2
    padding_dim2 = (dim2_new-dim2)/2
    padding_dim3 = (dim3_new-dim3)/2

    #Calculate padding in each side by taking decimal values into account
    #Dim1
    if padding_dim1 > 0:
        if np.floor(padding_dim1) == padding_dim1:
            pad1 = (int(padding_dim1),int(padding_dim1))
        else:
            pad1 = (int(np.floor(padding_dim1)),int(np.floor(padding_dim1)+1))
    else:
        pad1 = (0,0)
    #Dim2
    if padding_dim2 > 0:
        if np.floor(padding_dim2) == padding_dim2:
            pad2 = (int(padding_dim2),int(padding_dim2))
        else:
            pad2 = (int(np.floor(padding_dim2)),int(np.floor(padding_dim2)+1))
    else:
        pad2 = (0,0)
    #Dim3
    if padding_dim3 > 0:
        if np.floor(padding_dim3) == padding_dim3:
            pad3 = (int(padding_dim3),int(padding_dim3))
        else:
            pad3 = (int(np.floor(padding_dim3)),int(np.floor(padding_dim3)+1))
    else:
        pad3 = (0,0)

    #Doing padding
    data_img=np.pad(data_img, (pad1, pad2, pad3), constant_values = pad_value)

    #Change centroids coordinates to fit the padding
    for i in range(1,len(ctd_resampled_reoriented)):
        ctd_resampled_reoriented[i][1] = ctd_resampled_reoriented[i][1] - x_range[0] + pad1[0] #Husk også at ændre det længere oppe i restrictions! Har jeg gjort tror jeg?
        ctd_resampled_reoriented[i][2] = ctd_resampled_reoriented[i][2] - y_range[0] + pad2[0] #Husk også at ændre det længere oppe i restrictions! Har jeg gjort tror jeg?
        ctd_resampled_reoriented[i][3] = ctd_resampled_reoriented[i][3] - z_range[0] + pad3[0] #Husk også at ændre det længere oppe i restrictions! Har jeg gjort tror jeg?

    #SAVE RESTRCTIONS FILE for later convertion! Both take padding and cropping into account!
    x_convert = [- x_range[0] + pad1[0] ,- x_range[0] + pad1[0] + dim1]
    y_convert = [- y_range[0] + pad2[0] ,- y_range[0] + pad2[0] + dim2]
    z_convert = [- z_range[0] + pad3[0] ,- z_range[0] + pad3[0] + dim3] #z_range[0] will be zero, if you use COM for cropping instead of bounding box
    restrictions.update({subject: (x_convert[0],x_convert[1],y_convert[0],y_convert[1],z_convert[0],z_convert[1])})

    #Change hounsfield units
    if HU_cutoff == 1:
        data_img[data_img<HU_range_cutoff[0]] = HU_range_cutoff[0]
        data_img[data_img>HU_range_cutoff[1]] = HU_range_cutoff[1]

    #Gaussian smoothing
    data_img = gaussian_filter(data_img, sigma=0.75/vs[0]) #3/8 er bedre. 0.75/2 står der basically nu

    #Normalise houndsfield units
    if normalize_HU == 1:
        data_img = (HU_range_normalize[1]-HU_range_normalize[0])*(data_img - data_img.min()) / (data_img.max() - data_img.min()) + HU_range_normalize[0]
    
    #Save as nifti-file
    img_preprocessed = nib.Nifti1Image(data_img, img_resampled_reoriented.affine)

    print("DIMENSIONS")
    print(data_img.shape)
    
    #show_slices_dim1(data_img, subject)

    # show_centroids_dim1(data_img,img_preprocessed.header,ctd_resampled_reoriented)
    # show_centroids_dim2(data_img,img_preprocessed.header,ctd_resampled_reoriented)

    #Save data
    img_path = os.path.join(Output_folder,'img') #Create output-folders if it does not exist
    ctd_path = os.path.join(Output_folder,'ctd') #Create output-folders if it does not exist
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    if not os.path.exists(ctd_path):
        os.makedirs(ctd_path)
    nib.save(img_preprocessed, os.path.join(Output_folder, img_path, subject+'_img.nii.gz'))
    save_centroids(ctd_resampled_reoriented, os.path.join(Output_folder, ctd_path, subject+'_ctd.json'))


#Save padding
if not os.path.exists(Padding_output_directory): #Create the directory if it does not exist
    os.makedirs(Padding_output_directory)
with open(os.path.join(Padding_output_directory,Padding_output_filename), 'wb') as f:
    pickle.dump(restrictions, f)





#SKRALD! DOES NOT WORK. Tror grunden er følgende: np.round(x_min).astype(int). Vi afrunder og sådan lidt. Og det ved new_bb_coordinates jo ikke noget om.
#Calculate new bounding box for fun. Take padding into account.  
# x_min = new_bb_coordinates[0]-x_convert[0]
# x_max = new_bb_coordinates[1]-x_convert[0]
# y_min = new_bb_coordinates[2]-y_convert[0]
# y_max = new_bb_coordinates[3]-y_convert[0]
# z_min = new_bb_coordinates[4]-z_convert[0]
# z_max = new_bb_coordinates[5]-z_convert[0]
# bb_coordinates = (x_min,x_max,y_min,y_max,z_min,z_max)
