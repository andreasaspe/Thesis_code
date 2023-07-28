import os
from os import listdir
import numpy as np
import nibabel as nib
from data_utilities import *
import pandas as pd
import pickle

#######################################################
#################### CONTROL PANEL ####################
#######################################################
#Define scans
all_scans = 1 #Set to 1 if you want to preprocess all scans
list_of_subjects = ['sub-verse521'] #List of subjects

#Don't touch
New_orientation = ('L', 'A', 'S')
dim1_new = 64
dim2_new = 64
dim3_new = 128
New_voxel_size = 8 # [mm]

#Define directories
dir_image_original = '/scratch/s174197/data/Verse20/Verse20_test_unpacked'

Padding_output_directory = '/scratch/s174197/data/Verse20/Padding_specifications'
Padding_output_filename = 'pad_test'
#######################################################
#######################################################
#######################################################


#Define list of scans
if all_scans:
    all_subjects = []
    for filename in listdir(dir_image_original):
        subject = filename.split("_")[0]
        all_subjects.append(subject)
    all_subjects = np.unique(all_subjects)
else:
    all_subjects = list_of_subjects



restrictions = {}
for subject in all_subjects:
    print("\n\n")
    print("       SUBJECT: "+str(subject)+"\n")
    # LOAD FILES
    filename_img = [f for f in listdir(dir_image_original) if (f.startswith(subject) and f.endswith('img.nii.gz'))][0]
    img_nib = nib.load(os.path.join(dir_image_original,filename_img))

    #Rescale and reorient
    vs = (New_voxel_size,New_voxel_size,New_voxel_size)
    img_resampled = resample_nib(img_nib, voxel_spacing=vs, order=3)
    img_reoriented = reorient_to(img_resampled, axcodes_to=New_orientation)

    #Get data
    data_img = np.asanyarray(img_reoriented.dataobj, dtype=img_reoriented.dataobj.dtype)

    #Find dimensions
    dim1, dim2, dim3 = data_img.shape
    
    #Calculate padding in each side (volume should be centered)
    padding_dim1 = (dim1_new-dim1)/2
    padding_dim2 = (dim2_new-dim2)/2
    padding_dim3 = (dim3_new-dim3)/2
    
    #Calculate padding in each side by taking decimal values into account
    #Dim1
    if np.floor(padding_dim1) == padding_dim1:
        pad1 = (int(padding_dim1),int(padding_dim1))
    else:
        pad1 = (int(np.floor(padding_dim1)),int(np.floor(padding_dim1)+1))
    #Dim2
    if np.floor(padding_dim2) == padding_dim2:
        pad2 = (int(padding_dim2),int(padding_dim2))
    else:
        pad2 = (int(np.floor(padding_dim2)),int(np.floor(padding_dim2)+1))
    #Dim3
    if np.floor(padding_dim3) == padding_dim3:
        pad3 = (int(padding_dim3),int(padding_dim3))
    else:
        pad3 = (int(np.floor(padding_dim3)),int(np.floor(padding_dim3)+1))

    restrictions.update({subject: (pad1[0] , pad1[0]+dim1   ,   pad2[0] , pad2[0]+dim2   ,   pad3[0] , pad3[0]+dim3)})

# Create the directory if it does not exist
if not os.path.exists(Padding_output_directory):
    os.makedirs(Padding_output_directory)

with open(os.path.join(Padding_output_directory,Padding_output_filename), 'wb') as f:
    pickle.dump(restrictions, f)
