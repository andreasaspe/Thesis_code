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



#######################################################
#################### CONTROL PANEL ####################
#######################################################
#Define scans
all_scans = 0 #Set to 1 if you want to preprocess all scans
list_of_subjects = ['VERTEBRAE_HEALTHY_0001_SERIES0010'] #List of subjects
with open("E:\Andreas_s174197\Thesis\My_code\Other_scripts\list_of_subjects", "rb") as fp:   # Unpickling
    list_of_subjects = pickle.load(fp)


#Define directories
dir_data = r'F:\DTU-Vertebra-1\NIFTI' #'/zhome/bb/f/127616/Documents/Thesis/Rawdata_training'
Output_folder = r'E:\Andreas_s174197\data_RH\SpineLocalisation\data_prep' #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data' #"/Users/andreasaspe/Documents/Data/Preprocessed_data"
Padding_output_directory = 'E:/Andreas_s174197/data_RH/SpineLocalisation/Padding_specifications'
Padding_output_filename = 'pad'

#Define rescale and reorientation parameters
New_orientation = ('L', 'A', 'S')
New_voxel_size = 8 # [mm]

#Preprocessing
normalize_HU = 1 #Set to 1 if you want to normalize to the below range
HU_range_normalize = [-1, 1]
HU_cutoff = 1 #Set to 1 if you want to cut_off_HU. Define below:
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

    #LOAD IMAGE
    filename_img = [f for f in listdir(dir_data) if f.startswith(subject)][0]
    img_nib = nib.load(os.path.join(dir_data,filename_img))

    #RESAMPLE AND REORIENT
    vs = (New_voxel_size,New_voxel_size,New_voxel_size)
    #Image
    img_resampled = resample_nib(img_nib, voxel_spacing=vs, order=3)
    img_resampled_reoriented = reorient_to(img_resampled, axcodes_to=New_orientation)

    
    #Save data
    img_path = os.path.join(Output_folder,'img') #Create output-folders if it does not exist
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    nib.save(img_resampled_reoriented, os.path.join(Output_folder, img_path, subject+'-img.nii.gz'))