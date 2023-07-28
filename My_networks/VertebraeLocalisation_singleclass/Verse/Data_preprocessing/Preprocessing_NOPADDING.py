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
from my_data_utils import *
from my_plotting_functions import *

#######################################################
#################### CONTROL PANEL ####################
#######################################################
data_type = 'test'

#Define scans
all_scans = 0 #Set to 1 if you want to preprocess all scans
with open("/home/s174197/Thesis/MY_CODE/Other_scripts/list_of_subjects_"+data_type+"_VERSE", "rb") as fp:   # Unpickling
    list_of_subjects = pickle.load(fp)

#Define directories


### GPU CLUSTER ###
#Raw data
dir_data = '/scratch/s174197/data/Verse20/Verse20_'+data_type+'_unpacked' #'/scratch/s174197/data/Verse20/Verse20_test_unpacked' #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_test_unpacked'
#Output folders
Output_folder = '/scratch/s174197/data/Verse20/VertebraeLocalisation2/Verse20_'+data_type+'_prep_NOPADDING' #'/scratch/s174197/data/Verse20/Verse20_test_prep' #'/Users/andreasaspe/Documents/Data/Verse20_training_prep' #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data' #"/Users/andreasaspe/Documents/Data/Preprocessed_data"
#Før stod der: Output_folder_heatmaps. Det var noget pjat. Skal jeg ikke bruge til noget. Heller ikke i det originale

### MAC ### ?
# dir_heatmap = '/Users/andreasaspe/Documents/Data/Verse20/SpineLocalisation/Verse20_training_heatmaps' #'/scratch/s174197/data/Verse20/Verse20_test_predictions'  #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_test_predictions' #'/scratch/s174197/data/Verse20/Verse20_test_unpacked' #'/scratch/s174197/data/Verse20/Verse20_test_unpacked' #'/Users/andreasaspe/Documents/Data/Verse20_training' #r'C:\Users\PC\Documents\Andreas_s174197\dataset-verse20training_unpacked' #'/zhome/bb/f/127616/Documents/Thesis/Rawdata_training'
# dir_data = '/Users/andreasaspe/Documents/Data/Verse20/Verse20_training_unpacked'
# Output_folder = '/Users/andreasaspe/Documents/Data/VertebraeLocalisation/ONLYONESAMPLE_Verse20_training_prep' #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data' #"/Users/andreasaspe/Documents/Data/Preprocessed_data"
# dir_padding_specifications = '/Users/andreasaspe/Documents/Data/Verse20/SpineLocalisation/Padding_specifications/pad_training'

#Define rescale and reorientation parameters
New_orientation = ('L', 'A', 'S')
New_voxel_size = 2 # [mm]

#Preprocessing
HU_range_normalize = [-1, 1]
HU_range_cutoff = [-200, 1000] #Define HU range. Only takes effekt if HU_cutoff is set to 1
new_dim = (96,96,128)
#######################################################
#######################################################
#######################################################


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


for subject in tqdm(all_subjects):

    ctd_overview = []

    print("\n\n")
    print("       SUBJECT: "+str(subject)+"\n")

    #LOAD CENTROIDS
    filename_ctd = [f for f in listdir(dir_data) if (f.startswith(subject) and f.endswith('json'))][0]
    ctd_list = load_centroids(os.path.join(dir_data,filename_ctd))

    #LOAD original IMAGE
    filename_img = [f for f in listdir(dir_data) if (f.startswith(subject) and f.endswith('img.nii.gz'))][0]
    img_nib = nib.load(os.path.join(dir_data,filename_img))

    ##### START PREPROCESSING #####
    #Get info
    zooms = img_nib.header.get_zooms() #Voxel sizes
    axs_code = nio.ornt2axcodes(nio.io_orientation(img_nib.affine)) #Image orientation
    ctd_code = ctd_list[0] #Centroid orientation
    data_shape_voxels = img_nib.header.get_data_shape() #Shape of data
    data_shape_mm = np.array(data_shape_voxels)*np.array(zooms) #Data measures
    data_type = img_nib.header.get_data_dtype() #Data type

    #Get data
    data_img = np.asanyarray(img_nib.dataobj, dtype=img_nib.dataobj.dtype)

    # zooms = img_nib.header.get_zooms()
    # gap = 30
    # offset_z = int(np.round(gap/zooms[2]))
    # for ctd in ctd_list[1:]:
    #     if ctd[0] == 17:
    #         gap = 30
    #         z_coordinate = int(np.round(ctd[3]))
    #         data_img = data_img[:,:,:z_coordinate+offset_z]

    #Gaussian smoothing
    sigma = [0.75/zooms[0],0.75/zooms[1],0.75/zooms[2]]
    data_img = gaussian_filter(data_img, sigma=sigma)
    #Save as Nifti file
    img_nib = nib.Nifti1Image(data_img, img_nib.affine)

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

    #Change hounsfield units
    data_img[data_img<HU_range_cutoff[0]] = HU_range_cutoff[0]
    data_img[data_img>HU_range_cutoff[1]] = HU_range_cutoff[1]

    #Normalize HU
    min_value = np.min(data_img)
    max_value = np.max(data_img)
    data_img = (HU_range_normalize[1]-HU_range_normalize[0])*(data_img - min_value) / (max_value - min_value) + HU_range_normalize[0]
    
    #Convert to as nifti-file
    img_preprocessed = nib.Nifti1Image(data_img, img_resampled_reoriented.affine)

    #Save data
    img_path = os.path.join(Output_folder,'img') #Create output-folders if it does not exist
    ctd_path = os.path.join(Output_folder,'ctd') #Create output-folders if it does not exist
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    if not os.path.exists(ctd_path):
        os.makedirs(ctd_path)
    nib.save(img_preprocessed, os.path.join(Output_folder, img_path, subject+'_img.nii.gz'))
    save_centroids(ctd_resampled_reoriented, os.path.join(Output_folder, ctd_path, subject+'_ctd.json'))