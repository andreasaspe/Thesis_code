import os
import nibabel as nib
import nibabel.orientations as nio
import sys
#sys.path.append('E:/Andreas_s174197/Thesis/MY_CODE/utils')
from data_utilities import *
# from nibabel.affines import apply_affine
# import numpy.linalg as npl
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import pickle
from my_plotting_functions import *



#######################################################
#################### CONTROL PANEL ####################
#######################################################
#Define scans
all_scans = 0 #SET TO ZERO!! We want to process list_of_subject!!!!!!!!
# list_of_subjects = ['VERTEBRAE_FRACTURE_0280_SERIES0007'] #List of subjects
with open("E:\s174197\Thesis\My_code\Other_scripts\list_of_subjects", "rb") as fp:   # Unpickling, list_of_subjects_FRACTURE lidt forkert!
    list_of_subjects = pickle.load(fp)


#Define directories
dir_data = r'G:\DTU-Vertebra-1\NIFTI' #'/zhome/bb/f/127616/Documents/Thesis/Rawdata_training'
Output_folder = r'E:\s174197\data_RH\SpineLocalisation_newaffine\data_prep' #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data' #"/Users/andreasaspe/Documents/Data/Preprocessed_data"
# Padding_output_directory = 'E:/s174197/data_RH/SpineLocalisation/Padding_specifications'
# Padding_output_filename = 'pad'

#Define rescale and reorientation parameters
New_orientation = ('L', 'A', 'S')
New_voxel_size = 8 # [mm]

#Preprocessing
HU_range_normalize = [-1, 1]
HU_range_cutoff = [-200, 1000] #Define HU range. Only takes effekt if HU_cutoff is set to 1
pad_value = -1 # #Put in number or the string 'minimum' for padding with the minimum value in volume
dim1_new = 64
dim2_new = 64
dim3_new = 128
#######################################################
#######################################################
#######################################################

#Initialising list and dictionaries to save data on the way
dim1_list = []
dim2_list = []
dim3_list = []
# restrictions = {}


#Define list of scans
if all_scans:
    all_subjects = []
    for filename in listdir(dir_data):
        subject = filename.split(".")[0]
        #if subject.find('verse') != -1: #PLOTTER KUN VERSE. IKKE GL
        all_subjects.append(subject)
    all_subjects = np.unique(all_subjects)
    #Sorterer fil '.DS' fra
    all_subjects = all_subjects[all_subjects != '.DS']
else:
    all_subjects = list_of_subjects


#Create output-folders if it does not exist
img_path = os.path.join(Output_folder,'img')
if not os.path.exists(img_path):
    os.makedirs(img_path)

for subject in tqdm(all_subjects):
    try:
        print(subject)
    
        #LOAD IMAGE
        filename_img = [f for f in listdir(dir_data) if f.startswith(subject)][0]
        img_nib = nib.load(os.path.join(dir_data,filename_img))
    
        #Get info
        zooms = img_nib.header.get_zooms() #Voxel sizes
        axs_code = nio.ornt2axcodes(nio.io_orientation(img_nib.affine)) #Image orientation
        data_shape_voxels = img_nib.header.get_data_shape() #Shape of data
        data_shape_mm = np.array(data_shape_voxels)*np.array(zooms) #Data measures
        data_type = img_nib.header.get_data_dtype() #Data type
        
        if axs_code[0] in ['L','R']:
            LR = 0
        if axs_code[0] in ['A','P']:
            AP = 0
        if axs_code[0] in ['S','I']:
            SI = 0
    
        if axs_code[1] in ['L','R']:
            LR = 1
        if axs_code[1] in ['A','P']:
            AP = 1
        if axs_code[1] in ['S','I']:
            SI = 1
    
        if axs_code[2] in ['L','R']:
            LR = 2
        if axs_code[2] in ['A','P']:
            AP = 2
        if axs_code[2] in ['S','I']:
            SI = 2
        
        dim1_list_new = data_shape_mm[LR]
        dim2_list_new = data_shape_mm[AP]
        dim3_list_new = data_shape_mm[SI]
        dim1_list.append(dim1_list_new)
        dim2_list.append(dim2_list_new)
        dim3_list.append(dim3_list_new)
    
        if dim1_list_new > 512:
            print(str(subject) + " is too big in dimension 1. Size is "+ str(dim1_list_new))
            continue
        if dim2_list_new > 512:
            print(str(subject) + " is too big in dimension 2. Size is "+ str(dim2_list_new))
            continue
        if dim3_list_new > 1024:
            print(str(subject) + " is too big in dimension 3. Size is "+ str(dim3_list_new))
            continue
        
        # continue
    
        #Print info
        # print("Before any preprocessing:")
        # print('img orientation: {}'.format(axs_code))
        # print('img data shape in voxels: {}'.format(data_shape_voxels))
        # print('img data shape in mm: {}'.format(data_shape_mm))
        # print('img data type: {}'.format(data_type))
        # print("\n")
    
        #Gaussian smoothing!
        #Get data
        data_img = np.asanyarray(img_nib.dataobj, dtype=img_nib.dataobj.dtype)
        #Smooth
        sigma = [0.75/zooms[0],0.75/zooms[1],0.75/zooms[2]]
        data_img = gaussian_filter(data_img, sigma=sigma)
        #Save as Nifti file
        img_nib = nib.Nifti1Image(data_img, img_nib.affine)
    
        #RESAMPLE AND REORIENT
        vs = (New_voxel_size,New_voxel_size,New_voxel_size)
        #Image
        img_resampled = resample_nib(img_nib, voxel_spacing=vs, order=3)
        img_resampled_reoriented = reorient_to(img_resampled, axcodes_to=New_orientation)
    
        #Get info
        zooms = img_resampled_reoriented.header.get_zooms() #Voxel sizes
        axs_code = nio.ornt2axcodes(nio.io_orientation(img_resampled_reoriented.affine)) #Image orientation
        data_shape_voxels = img_resampled_reoriented.header.get_data_shape() #Shape of data
        data_shape_mm = np.array(data_shape_voxels)*np.array(zooms) #Data measures
        data_type = img_resampled_reoriented.header.get_data_dtype() #Data type

    
        #Print info
        # print("AFTER RESCALING AND REORIENTATION:")
        # print('img orientation: {}'.format(axs_code))
        # print('img data shape in voxels: {}'.format(data_shape_voxels))
        # print('img data shape in mm: {}'.format(data_shape_mm))
        # print('img data type: {}'.format(data_type))
        # print("\n")
    
        #Load data
        data_img = np.asanyarray(img_resampled_reoriented.dataobj, dtype=img_resampled_reoriented.dataobj.dtype)
        
        #Change hounsfield units
        data_img[data_img<HU_range_cutoff[0]] = HU_range_cutoff[0]
        data_img[data_img>HU_range_cutoff[1]] = HU_range_cutoff[1]
    
        #Normalize HU
        data_min = np.min(data_img)
        data_max = np.max(data_img)
        data_img = (HU_range_normalize[1]-HU_range_normalize[0])*(data_img - data_min) / (data_max - data_min) + HU_range_normalize[0]
    

    
        # #BORDER PADDING
        # #Find dimensions
        # dim1, dim2, dim3 = data_img.shape
        
        # #Calculate padding in each side (volume should be centered)
        # padding_dim1 = (dim1_new-dim1)/2
        # padding_dim2 = (dim2_new-dim2)/2
        # padding_dim3 = (dim3_new-dim3)/2
               
        # #Calculate padding in each side by taking decimal values into account
        # #Dim1
        # if np.floor(padding_dim1) == padding_dim1:
        #     pad1 = (int(padding_dim1),int(padding_dim1))
        # else:
        #     pad1 = (int(np.floor(padding_dim1)),int(np.floor(padding_dim1)+1))
        # #Dim2
        # if np.floor(padding_dim2) == padding_dim2:
        #     pad2 = (int(padding_dim2),int(padding_dim2))
        # else:
        #     pad2 = (int(np.floor(padding_dim2)),int(np.floor(padding_dim2)+1))
        # #Dim3
        # if np.floor(padding_dim3) == padding_dim3:
        #     pad3 = (int(padding_dim3),int(padding_dim3))
        # else:
        #     pad3 = (int(np.floor(padding_dim3)),int(np.floor(padding_dim3)+1))
        
        # #Save padding specifications
        # restrictions.update({subject: (pad1[0] , pad1[0]+dim1   ,   pad2[0] , pad2[0]+dim2   ,   pad3[0] , pad3[0]+dim3)})
    
        # #Doing padding
        # data_img=np.pad(data_img, (pad1, pad2, pad3), constant_values = pad_value)
        # #Find new dimensions
        # dim1, dim2, dim3 = data_img.shape
        
        new_affine = np.array([[8,0,0,0],[0,8,0,0],[0,0,8,0],[0,0,0,1]])
    
    
        #Define as new Nifti-file
        img_preprocessed = nib.Nifti1Image(data_img, new_affine) #img_resampled_reoriented.affine)
    
        #Save
        nib.save(img_preprocessed, os.path.join(Output_folder, img_path, subject+'-img.nii.gz'))
        
        #Save padding info
        # # Create the directory if it does not exist
        # if not os.path.exists(Padding_output_directory):
        #     os.makedirs(Padding_output_directory)
        # #Save
        # with open(os.path.join(Padding_output_directory,Padding_output_filename), 'wb') as f:
        #     pickle.dump(restrictions, f)
    except:
        print("FAIL!! THE FOLLOWING SUBJECT FAILS:")
        print(subject)


print("Done")
#Max
print("Dim1 max :"+str(np.max(dim1_list)))
print("Dim2 max :"+str(np.max(dim2_list)))
print("Dim3 max :"+str(np.max(dim3_list)))
#Min
print("Dim1 min :"+str(np.min(dim1_list)))
print("Dim2 min :"+str(np.min(dim2_list)))
print("Dim3 min :"+str(np.min(dim3_list)))
#Mean
print("Dim1 mean :"+str(np.mean(dim1_list)))
print("Dim2 mean :"+str(np.mean(dim2_list)))
print("Dim3 mean :"+str(np.mean(dim3_list)))