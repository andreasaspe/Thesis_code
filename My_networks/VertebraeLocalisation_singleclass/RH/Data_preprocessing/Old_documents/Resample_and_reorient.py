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
all_scans = 1 #Set to 1 if you want to preprocess all scans
list_of_subjects = ['VERTEBRAE_HEALTHY_0001_SERIES0010'] #List of subjects
with open("E:\Andreas_s174197\Thesis\My_code\Other_scripts\list_of_subjects", "rb") as fp:   # Unpickling
    list_of_subjects = pickle.load(fp)


#Define directories
dir_data = r'F:\DTU-Vertebra-1\NIFTI' #'/zhome/bb/f/127616/Documents/Thesis/Rawdata_training'
Output_folder = r'E:\Andreas_s174197\data_RH\data_prep_temp' #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data' #"/Users/andreasaspe/Documents/Data/Preprocessed_data"
Padding_output_directory = 'E:/Andreas_s174197/data_RH/Padding_specifications'
Padding_output_filename = 'pad_temp'

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
    filename_img = [f for f in listdir(dir_data) if (f.startswith(subject) and f.endswith('img.nii.gz'))][0]
    img_nib = nib.load(os.path.join(dir_data,filename_img))

    #LOAD MASK
    filename_msk = [f for f in listdir(dir_data) if (f.startswith(subject) and f.endswith('msk.nii.gz'))][0]
    msk_nib = nib.load(os.path.join(dir_data,filename_msk))

    #LOAD CENTROIDS
    filename_ctd = [f for f in listdir(dir_data) if (f.startswith(subject) and f.endswith('json'))][0]
    ctd_list = load_centroids(os.path.join(os.path.join(dir_data,filename_ctd)))


    #RESAMPLE AND REORIENT
    vs = (New_voxel_size,New_voxel_size,New_voxel_size)
    #Image
    img_resampled = resample_nib(img_nib, voxel_spacing=vs, order=3)
    img_resampled_reoriented = reorient_to(img_resampled, axcodes_to=New_orientation)
    #Mask
    msk_resampled = resample_nib(msk_nib, voxel_spacing=vs, order=0) # or resample based on img: resample_mask_to(msk_nib, img_iso)
    msk_resampled_reoriented = reorient_to(msk_resampled, axcodes_to=New_orientation)
    #Centroids
    ctd_resampled = rescale_centroids(ctd_list, img_nib, vs)
    ctd_resampled_reoriented = reorient_centroids_to(ctd_resampled, img_resampled_reoriented)

    # #Load data
    # data_img = np.asanyarray(img_resampled_reoriented.dataobj, dtype=img_resampled_reoriented.dataobj.dtype)
    # data_msk = np.asanyarray(msk_resampled_reoriented.dataobj, dtype=msk_resampled_reoriented.dataobj.dtype)

    # #Change hounsfield units
    # if HU_cutoff == 1:
    #     data_img[data_img<HU_range_cutoff[0]] = HU_range_cutoff[0]
    #     data_img[data_img>HU_range_cutoff[1]] = HU_range_cutoff[1]

    # # #Gaussian smoothing
    # # data_img = gaussian_filter(data_img, sigma=0.75/vs[0]) #3/8 er bedre. 0.75/2 står der basically nu

    # #Normalise houndsfield units
    # if normalize_HU == 1:
    #     data_img = (HU_range_normalize[1]-HU_range_normalize[0])*(data_img - data_img.min()) / (data_img.max() - data_img.min()) + HU_range_normalize[0]
    
    # #Save as nifti-file
    # img_preprocessed = nib.Nifti1Image(data_img, img_resampled_reoriented.affine)
    # msk_preprocessed = nib.Nifti1Image(data_msk, msk_resampled_reoriented.affine)
    
    #Save data
    img_path = os.path.join(Output_folder,'img') #Create output-folders if it does not exist
    msk_path = os.path.join(Output_folder,'msk') #Create output-folders if it does not exist
    ctd_path = os.path.join(Output_folder,'ctd') #Create output-folders if it does not exist
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    if not os.path.exists(msk_path):
        os.makedirs(msk_path)
    if not os.path.exists(ctd_path):
        os.makedirs(ctd_path)
    nib.save(img_resampled_reoriented, os.path.join(Output_folder, img_path, subject+'-img.nii.gz'))
    nib.save(msk_resampled_reoriented, os.path.join(Output_folder, msk_path, subject+'-msk.nii.gz'))
    save_centroids(ctd_resampled_reoriented, os.path.join(Output_folder, ctd_path, subject+'-ctd.json'))