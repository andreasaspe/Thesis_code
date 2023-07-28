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
data_type = 'test'

all_scans = 0 #Set to 1 if you want to preprocess all scans
with open("/home/s174197/Thesis/MY_CODE/Other_scripts/list_of_subjects_"+data_type+"_VERSE", "rb") as fp:   # Unpickling
    list_of_subjects = pickle.load(fp)
# list_of_subjects = ['sub-verse507'] #List of subjects 521

#Define rescale and reorientation parameters
New_orientation = ('L', 'A', 'S')
New_voxel_size = 8 # [mm]

#If you want to print output
Print_info = 0 #Set to 1 to print a lot of info on each scan.

#Define preprocessing details and printing output[s
HU_range_normalize = [-1, 1]
HU_range_cutoff = [-200, 1000] #Define HU range. Only takes effekt if HU_cutoff is set to 1
pad_value = -1 # #Put in number or the string 'minimum' for padding with the minimum value in volume
new_dim  = (64,64,128)


#Define directories
#Cluster
dir_data = '/scratch/s174197/data/Verse20/Verse20_'+data_type+'_unpacked' #'/scratch/s174197/data/Verse20/Verse20_training_unpacked' #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_test_unpacked' #'/scratch/s174197/data/Verse20/Verse20_test_unpacked' #'/scratch/s174197/data/Verse20/Verse20_test_unpacked' #'/Users/andreasaspe/Documents/Data/Verse20_training' #r'C:\Users\PC\Documents\Andreas_s174197\dataset-verse20training_unpacked' #'/zhome/bb/f/127616/Documents/Thesis/Rawdata_training'
Output_folder = '/scratch/s174197/data/Verse20/SpineLocalisation/Verse20_'+data_type+'_prep_NOPADDING' #'/scratch/s174197/data/Verse20/Verse20_test_prep' #'/Users/andreasaspe/Documents/Data/Verse20_training_prep' #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data' #"/Users/andreasaspe/Documents/Data/Preprocessed_data"
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

#FOR LOOP START
for subject in tqdm(all_subjects):
    print("\n\n")
    print("       SUBJECT: "+str(subject)+"\n")
    # LOAD FILES
    # Define file names
    filename_img = [f for f in listdir(dir_data) if (f.startswith(subject) and f.endswith('img.nii.gz'))][0]
    filename_ctd = [f for f in listdir(dir_data) if (f.startswith(subject) and f.endswith('json'))][0]
    # Load files
    img_nib = nib.load(os.path.join(dir_data,filename_img))
    ctd_list = load_centroids(os.path.join(os.path.join(dir_data,filename_ctd)))


    #Get info
    zooms = img_nib.header.get_zooms() #Voxel sizes
    axs_code = nio.ornt2axcodes(nio.io_orientation(img_nib.affine)) #Image orientation
    ctd_code = ctd_list[0] #Centroid orientation
    data_shape_voxels = img_nib.header.get_data_shape() #Shape of data
    data_shape_mm = np.array(data_shape_voxels)*np.array(zooms) #Data measures
    data_type = img_nib.header.get_data_dtype() #Data type

    #Print info
    print("Before any preprocessing:")
    print('img orientation: {}'.format(axs_code))
    print('centroids orientation: {}'.format(ctd_code))
    print('img data shape in voxels: {}'.format(data_shape_voxels))
    print('img data shape in mm: {}'.format(data_shape_mm))
    print('img data type: {}'.format(data_type))
    print("\n")

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

    #Resample and reorient
    vs = (New_voxel_size,New_voxel_size,New_voxel_size)
    img_resampled = resample_nib(img_nib, voxel_spacing=vs, order=3)
    ctd_resampled = rescale_centroids(ctd_list, img_nib, vs)
    img_resampled_reoriented = reorient_to(img_resampled, axcodes_to=New_orientation)
    ctd_resampled_reoriented = reorient_centroids_to(ctd_resampled, img_resampled_reoriented)


    #Get info
    zooms = img_resampled_reoriented.header.get_zooms() #Voxel sizes
    axs_code = nio.ornt2axcodes(nio.io_orientation(img_resampled_reoriented.affine)) #Image orientation
    ctd_code = ctd_resampled_reoriented[0] #Centroid orientation
    data_shape_voxels = img_resampled_reoriented.header.get_data_shape() #Shape of data
    data_shape_mm = np.array(data_shape_voxels)*np.array(zooms) #Data measures
    data_type = img_resampled_reoriented.header.get_data_dtype() #Data type
    #Print info
    print("AFTER RESCALING AND REORIENTATION:")
    print('img orientation: {}'.format(axs_code))
    print('centroids orientation: {}'.format(ctd_code))
    print('img data shape in voxels: {}'.format(data_shape_voxels))
    print('img data shape in mm: {}'.format(data_shape_mm))
    print('img data type: {}'.format(data_type))
    print("\n")
        

    #Load data
    data_img = np.asanyarray(img_resampled_reoriented.dataobj, dtype=img_resampled_reoriented.dataobj.dtype)
    
    #Change hounsfield units
    data_img[data_img<HU_range_cutoff[0]] = HU_range_cutoff[0]
    data_img[data_img>HU_range_cutoff[1]] = HU_range_cutoff[1]

    #Normalize HU
    data_img = (HU_range_normalize[1]-HU_range_normalize[0])*(data_img - data_img.min()) / (data_img.max() - data_img.min()) + HU_range_normalize[0]

    #Define as new Nifti-file
    img_preprocessed = nib.Nifti1Image(data_img, img_resampled_reoriented.affine)
    
    #Save image and centroids
    img_path = os.path.join(Output_folder,'img')
    ctd_path = os.path.join(Output_folder,'ctd')
    #Create output-folders if it does not exist
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    if not os.path.exists(ctd_path):
        os.makedirs(ctd_path)
    #Save
    nib.save(img_preprocessed, os.path.join(Output_folder, img_path, subject+'_img.nii.gz'))
    save_centroids(ctd_resampled_reoriented, os.path.join(Output_folder, ctd_path, subject+'_ctd.json'))


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