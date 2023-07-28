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
#Define scans
all_scans = 1 #Set to 1 if you want to preprocess all scans
list_of_subjects = ['sub-verse500'] #sub-verse820, 'sub-verse824' is edge case. Too short y. List of subjects 521, 


#Define directories


### GPU CLUSTER ###
#Type
data_type = 'training' #Training, validation or test
#Raw data
dir_data = '/scratch/s174197/data/Verse20/Verse20_'+data_type+'_unpacked' #'/scratch/s174197/data/Verse20/Verse20_test_unpacked' #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_test_unpacked'
#SpineLocalisation folders
dir_heatmap = '/scratch/s174197/data/Verse20/SpineLocalisation/Verse20_'+data_type+'_heatmaps' #'/Users/andreasaspe/Documents/Data/Verse20/SpineLocalisation/Verse20_test_predictions' #'/scratch/s174197/data/Verse20/Verse20_test_predictions'  #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_test_predictions' #'/scratch/s174197/data/Verse20/Verse20_test_unpacked' #'/scratch/s174197/data/Verse20/Verse20_test_unpacked' #'/Users/andreasaspe/Documents/Data/Verse20_training' #r'C:\Users\PC\Documents\Andreas_s174197\dataset-verse20training_unpacked' #'/zhome/bb/f/127616/Documents/Thesis/Rawdata_training'
dir_padding_specifications = '/scratch/s174197/data/Verse20/SpineLocalisation/Padding_specifications/pad_'+data_type
#Output folders
Output_folder = '/scratch/s174197/data/Verse20/VertebraeLocalisation2/Verse20_'+data_type+'_prep_alldata' #'/scratch/s174197/data/Verse20/Verse20_test_prep' #'/Users/andreasaspe/Documents/Data/Verse20_training_prep' #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data' #"/Users/andreasaspe/Documents/Data/Preprocessed_data"
Output_folder_heatmaps = '/scratch/s174197/data/Verse20/VertebraeLocalisation2/Verse20_'+data_type+'_heatmaps_alldata'
Padding_output_directory = '/scratch/s174197/data/Verse20/VertebraeLocalisation2/Padding_specifications'
Padding_output_filename = 'pad_'+data_type+'_alldata'

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

#Load padding_specifications
with open(dir_padding_specifications, 'rb') as f:
    padding_specifications = pickle.load(f) 

restrictions_dict = {}

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
    heatmap_nib = nib.load(heatmap_file_dir)
    heatmap_data = np.asanyarray(heatmap_nib.dataobj, dtype=np.float32)

    #LOAD original image IMAGE
    filename_img = [f for f in listdir(dir_data) if (f.startswith(subject) and f.endswith('img.nii.gz'))][0]
    img_nib = nib.load(os.path.join(dir_data,filename_img))

    #FIND BOUNDING BOX
    old_restrictions = padding_specifications[subject]
    old_zooms = (8,8,8)
    new_zooms = img_nib.header.get_zooms()
    bb_coordinates, COM = BoundingBox(heatmap_data,old_restrictions)
    original_bb_coordinates, original_COM = RescaleBoundingBox(new_zooms,old_zooms,bb_coordinates,COM,old_restrictions)

    new_zooms = (2,2,2)
    old_zooms = img_nib.header.get_zooms()
    new_bb_coordinates, new_COM = RescaleBoundingBox(new_zooms,old_zooms,original_bb_coordinates,original_COM)


    ##### START PREPROCESSING #####
    #Get info
    zooms = img_nib.header.get_zooms() #Voxel sizes
    axs_code = nio.ornt2axcodes(nio.io_orientation(img_nib.affine)) #Image orientation
    ctd_code = ctd_list[0] #Centroid orientation
    data_shape_voxels = img_nib.header.get_data_shape() #Shape of data
    data_shape_mm = np.array(data_shape_voxels)*np.array(zooms) #Data measures
    data_type = img_nib.header.get_data_dtype() #Data type

    #Gaussian smoothing
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
    #z_range = [z_min,z_max]
    z_range = [0,0]
    data_img = data_img[x_range[0]:x_range[1],y_range[0]:y_range[1],:] #z_range[0]:z_range[1]
    
    #BORDER PADDING if necessary
    data_img, restrictions = center_and_pad(data=data_img, new_dim=new_dim, pad_value=-1)
    x_min_restrict, _, y_min_restrict, _, z_min_restrict, _ = restrictions #For padding

    dim1,dim2,dim3 = data_img.shape

    #SAVE RESTRCTIONS FILE for later convertion! Both take padding and cropping into account!
    x_convert = [x_min_restrict - x_range[0],x_min_restrict - x_range[0] + dim1]
    y_convert = [y_min_restrict - y_range[0],y_min_restrict - y_range[0] + dim2]
    z_convert = [z_min_restrict - z_range[0],z_min_restrict - z_range[0]+ dim3] #z_range[0] will be zero, if you use COM for cropping instead of bounding box
    restrictions_dict.update({subject: (x_convert[0],x_convert[1],y_convert[0],y_convert[1],z_convert[0],z_convert[1])})

    #Apply transformation to centroids
    for i in range(len(ctd_resampled_reoriented)-1):
        ctd_resampled_reoriented[i+1][1] += x_convert[0]
        ctd_resampled_reoriented[i+1][2] += y_convert[0]
        ctd_resampled_reoriented[i+1][3] += z_convert[0]

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

#Save padding
if not os.path.exists(Padding_output_directory): #Create the directory if it does not exist
    os.makedirs(Padding_output_directory)
with open(os.path.join(Padding_output_directory,Padding_output_filename), 'wb') as f:
    pickle.dump(restrictions_dict, f)










    #Old ranges:
    # x_range = [max(new_x_COM-48,0),min(new_x_COM+48,dim1)]
    # y_range = [max(new_y_COM-48,0),min(new_y_COM+48,dim2)]
    # z_range = [0,0]