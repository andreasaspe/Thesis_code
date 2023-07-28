

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
from copy import deepcopy
from tqdm import tqdm
#My functions
from my_data_utils import *
from my_plotting_functions import *



#######################################################
#################### CONTROL PANEL ####################
#######################################################
#Define scans
all_scans = 1 #Set to 1 if you want to preprocess all scans
list_of_subjects = ['sub-verse820'] #List of subjects 521
sigma = 5 #Parameter for heatmaps

#######################################################
#######################################################
#######################################################

#Define directories


### GPU CLUSTER ###
dir_data = '/scratch/s174197/data/Verse20/VertebraeSegmentation/Verse20_training_rescaled' #Overall folder. Defining specific below #'/scratch/s174197/data/Verse20/Verse20_test_unpacked' #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_test_unpacked'
#Outputs
Output_folder = '/scratch/s174197/data/Verse20/VertebraeSegmentation/Verse20_training_prep'
#Padding
Padding_output_directory = '/scratch/s174197/data/Verse20/VertebraeSegmentation/Padding_specifications'
Padding_output_filename = 'pad_training'

### mac ###
#dir_data = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeSegmentation/SUBSET_Verse20_training_rescaled' #Overall folder. Defining specific below #'/scratch/s174197/data/Verse20/Verse20_test_unpacked' #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_test_unpacked'
#Outputs
# Output_folder = '/scratch/s174197/data/Verse20/VertebraeSegmentation/Verse20_validation_prep'
# #Padding
# Padding_output_directory = '/scratch/s174197/data/Verse20/VertebraeSegmentation/Padding_specifications'
# Padding_output_filename = 'pad_validation'

#Define rescale and reorientation parameters
New_orientation = ('L', 'A', 'S')
New_voxel_size = 1 # [mm]

#Preprocessing
normalize_HU = 1 #Set to 1 if you want to normalize to the below range
HU_range_normalize = [-1, 1]
HU_cutoff = 1 #Set to 1 if you want to cut_off_HU. Define below:
HU_range_cutoff = [-200, 1000] #Define HU range. Only takes effekt if HU_cutoff is set to 1
#######################################################
#######################################################
#######################################################

img_dir = os.path.join(dir_data,'img')
msk_dir = os.path.join(dir_data,'msk')
ctd_dir = os.path.join(dir_data,'ctd')

#Define list of scans
if all_scans:
    all_subjects = []
    for filename in listdir(img_dir):
        subject = filename.split("_")[0]
        #if subject.find('verse') != -1: #PLOTTER KUN VERSE. IKKE GL
        all_subjects.append(subject)
    all_subjects = np.unique(all_subjects)
    #Sorterer fil '.DS' fra
    all_subjects = all_subjects[all_subjects != '.DS']
else:
    all_subjects = list_of_subjects

padding_specifications = {}


#Create folders for saving data
img_path = os.path.join(Output_folder,'img') #Create output-folders if it does not exist
heatmap_path = os.path.join(Output_folder,'heatmaps') #Create output-folders if it does not exist
msk_path = os.path.join(Output_folder,'msk') #Create output-folders if it does not exist
if not os.path.exists(img_path):
    os.makedirs(img_path)
if not os.path.exists(heatmap_path):
    os.makedirs(heatmap_path)
if not os.path.exists(msk_path):
    os.makedirs(msk_path)

for subject in tqdm(all_subjects):

    #LOAD IMAGE
    filename_img = [f for f in listdir(img_dir) if (f.startswith(subject) and f.endswith('img.nii.gz'))][0]
    img_nib = nib.load(os.path.join(img_dir,filename_img))

    #LOAD MASK
    filename_msk = [f for f in listdir(msk_dir) if (f.startswith(subject) and f.endswith('msk.nii.gz'))][0]
    msk_nib = nib.load(os.path.join(msk_dir,filename_msk))

    #LOAD CENTROIDS
    filename_ctd = [f for f in listdir(ctd_dir) if (f.startswith(subject) and f.endswith('json'))][0]
    ctd_list = load_centroids(os.path.join(os.path.join(ctd_dir,filename_ctd)))

    #Load data
    data_img = np.asanyarray(img_nib.dataobj, dtype=img_nib.dataobj.dtype)
    data_msk = np.asanyarray(msk_nib.dataobj, dtype=msk_nib.dataobj.dtype)

    #Crop image and mask based on centroids!
    for ctd in ctd_list[1:]:
        if 17 <= ctd[0] <= 24:
            x = np.round(ctd[1]).astype(int)
            y = np.round(ctd[2]).astype(int)
            z = np.round(ctd[3]).astype(int)

            x_start = x-64
            x_end = x+63
            y_start = y-64
            y_end = y+63
            z_start = z-48
            z_end = z+47

            #Dimensions
            dim1, dim2, dim3 = data_img.shape

            if x_start < 0:
                print("Shit x")
                x_start = 0
                x_end = 127
            if x_end > dim1 - 1:
                x_start = (dim1 - 1) - 127
                x_end = dim1 - 1

            #y-axis
            if y_start < 0:
                print("Shit y")
                y_start = 0
                y_end = 127
            if y_end > dim2 - 1:
                y_start = (dim2 - 1) - 127
                y_end = dim2 - 1

            #z-axis
            if z_start < 0:
                print("Shit z")
                z_start = 0
                z_end = 95
            if z_end > dim3 - 1:
                z_start = (dim3 - 1) - 95
                z_end = dim3 - 1

            #Crop data
            data_img_temp = deepcopy(data_img[x_start:x_end+1,y_start:y_end+1,z_start:z_end+1])
            data_msk_temp = deepcopy(data_msk[x_start:x_end+1,y_start:y_end+1,z_start:z_end+1])
            #Remove all other masks than the relevant one and convert to binary
            #data_msk_temp[ data_msk_temp!=ctd[0] ] = 0
            data_msk_temp = np.where(data_msk_temp == ctd[0],1,0)
            
            # show_slices_dim1(data_img_temp, subject)
            # show_mask_dim1(data_img_temp, data_msk_temp)

            
            subject_ID = subject + '-' + str(ctd[0])
            padding_specifications.update({subject_ID: (x_start,x_end,y_start,y_end,z_start,z_end)}) #Burde jeg sige minus her?

            #Change hounsfield units
            if HU_cutoff == 1:
                data_img_temp[data_img_temp<HU_range_cutoff[0]] = HU_range_cutoff[0]
                data_img_temp[data_img_temp>HU_range_cutoff[1]] = HU_range_cutoff[1]

            # #Gaussian smoothing
            # data_img = gaussian_filter(data_img, sigma=0.75/vs[0]) #3/8 er bedre. 0.75/2 står der basically nu

            #Normalise houndsfield units
            if normalize_HU == 1:
                data_img_temp = (HU_range_normalize[1]-HU_range_normalize[0])*(data_img_temp - data_img_temp.min()) / (data_img_temp.max() - data_img_temp.min()) + HU_range_normalize[0]

            #Generate heatmap
            origins = (x-x_start,y-y_start,z-z_start) #Convert for cropping
            meshgrid_dim = (128,128,96)
            heatmap = gaussian_kernel_3d_new(origins,meshgrid_dim,sigma)
            #Normalize
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

            #Define filenames and save data
            img_filename = subject_ID + "_img.npy" #Input
            heatmap_filename = subject_ID + "_heatmap.npy" #Input
            msk_filename = subject_ID + "_msk.npy" #Target
            np.save(os.path.join(img_path,img_filename), data_img_temp)
            np.save(os.path.join(heatmap_path,heatmap_filename), heatmap)
            np.save(os.path.join(msk_path,msk_filename), data_msk_temp)


#Save padding-directory
if not os.path.exists(Padding_output_directory): #Create the directory if it does not exist
    os.makedirs(Padding_output_directory)
with open(os.path.join(Padding_output_directory,Padding_output_filename), 'wb') as f:
    pickle.dump(padding_specifications, f)







    # img_filename = subject_ID + "_img.npy"
    # heatmap_filename = subject_ID + "_heatmap.npy"

    # # #Save data
    # # img_path = os.path.join(Output_folder,'img') #Create output-folders if it does not exist
    # # if not os.path.exists(img_path):
    # #     os.makedirs(img_path)

    # # np.save("array_list.npy", array_list)







#Append to list
           # save_data.update({subject_ID: (data_img_temp,data_msk_temp,heatmap)}) #Burde jeg sige minus her?