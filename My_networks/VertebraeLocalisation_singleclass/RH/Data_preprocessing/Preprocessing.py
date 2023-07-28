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
with open("E:\s174197\Thesis\My_code\Other_scripts\list_of_subjects", "rb") as fp:   # Unpickling
    list_of_subjects = pickle.load(fp)


#Define directories
dir_data = r'G:\DTU-Vertebra-1\NIFTI' #'/zhome/bb/f/127616/Documents/Thesis/Rawdata_training'
Output_folder = r'E:\s174197\data_RH\VertebraeLocalisation2_newaffine\data_prep' #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data' #"/Users/andreasaspe/Documents/Data/Preprocessed_data"

#Define rescale and reorientation parameters
New_orientation = ('L', 'A', 'S')
New_voxel_size = 2 # [mm]

#Preprocessing
HU_range_normalize = [-1, 1]
HU_range_cutoff = [-200, 1000] #Define HU range. Only takes effekt if HU_cutoff is set to 1
#######################################################
#######################################################
#######################################################


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
        
        new_affine = np.array([[2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,1]])

        #Define as new Nifti-file
        img_preprocessed = nib.Nifti1Image(data_img, new_affine) #img_resampled_reoriented.affine)
    
        #Save image
        img_path = os.path.join(Output_folder,'img')
    
        #Create output-folders if it does not exist
        if not os.path.exists(img_path):
            os.makedirs(img_path)
    
        #Save
        nib.save(img_preprocessed, os.path.join(Output_folder, img_path, subject+'-img.nii.gz'))
    except:
        print("FAIL!! THE FOLLOWING SUBJECT FAILS:")
        print(subject)