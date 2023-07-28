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
all_scans = 1 #Set to 1 if you want to preprocess all scans


#Define directories
#Cluster
data_type = 'test'
dir_data = '/scratch/s174197/data/Verse20/Verse20_'+data_type+'_unpacked' #'/scratch/s174197/data/Verse20/Verse20_training_unpacked' #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_test_unpacked' #'/scratch/s174197/data/Verse20/Verse20_test_unpacked' #'/scratch/s174197/data/Verse20/Verse20_test_unpacked' #'/Users/andreasaspe/Documents/Data/Verse20_training' #r'C:\Users\PC\Documents\Andreas_s174197\dataset-verse20training_unpacked' #'/zhome/bb/f/127616/Documents/Thesis/Rawdata_training'
Output_folder = '/home/s174197/Thesis/MY_CODE/Other_scripts'
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

compatible_subjects = []
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

    ctd_overview = []

    for ctd in ctd_list[1:]:
        if 17 <= ctd[0] <= 24:
            ctd_overview.append(ctd[0])

    if ctd_overview == []: #Hvis der ikke er nogen centroids i spændet.
        print("No centroids in range..")
        continue
    else:
        min_ctd = min(ctd_overview)
        max_ctd = max(ctd_overview)
        print("Yes, we have centroids. Range is: ["+str(min_ctd)+","+str(max_ctd)+"]")
        if max_ctd - min_ctd < 7:
            #print("Dropping it, because there are less than three centroids visible")
            print("Dropping it, because not all centroids are visible")
            continue

    compatible_subjects.append(subject)


# with open(os.path.join(Output_folder,"list_of_subjects_"+data_type+"_VERSE"), "wb") as fp:   #Pickling
#     pickle.dump(compatible_subjects, fp)

print(len(compatible_subjects))










#OLD VERSION
# #######################################################
# #################### CONTROL PANEL ####################
# #######################################################
# #Define scans
# all_scans = 1 #Set to 1 if you want to preprocess all scans


# #Define directories
# #Cluster
# data_type = 'test'
# dir_data = '/scratch/s174197/data/Verse20/Verse20_'+data_type+'_unpacked' #'/scratch/s174197/data/Verse20/Verse20_training_unpacked' #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_test_unpacked' #'/scratch/s174197/data/Verse20/Verse20_test_unpacked' #'/scratch/s174197/data/Verse20/Verse20_test_unpacked' #'/Users/andreasaspe/Documents/Data/Verse20_training' #r'C:\Users\PC\Documents\Andreas_s174197\dataset-verse20training_unpacked' #'/zhome/bb/f/127616/Documents/Thesis/Rawdata_training'
# Output_folder = '/home/s174197/Thesis/MY_CODE/Other_scripts'
# #######################################################
# #######################################################
# #######################################################

# #Define list of scans
# if all_scans:
#     all_subjects = []
#     for filename in listdir(dir_data):
#         subject = filename.split("_")[0]
#         #if subject.find('verse') != -1:
#         all_subjects.append(subject)
#     all_subjects = np.unique(all_subjects)
#     all_subjects = all_subjects[all_subjects != '.DS'] #Sorting out .DS
# else:
#     all_subjects = list_of_subjects




# #Initialising list and dictionaries to save data on the way
# dim1_list = []
# dim2_list = []
# dim3_list = []
# restrictions_dict = {}

# compatible_subjects = []
# idx = 0
# #FOR LOOP START
# for subject in tqdm(all_subjects):
#     idx+=1
#     print("\n\n")
#     print("       SUBJECT: "+str(subject)+"\n")
#     # LOAD FILES
#     # Define file names
#     filename_img = [f for f in listdir(dir_data) if (f.startswith(subject) and f.endswith('img.nii.gz'))][0]
#     filename_ctd = [f for f in listdir(dir_data) if (f.startswith(subject) and f.endswith('json'))][0]
#     # Load files
#     img_nib = nib.load(os.path.join(dir_data,filename_img))
#     ctd_list = load_centroids(os.path.join(os.path.join(dir_data,filename_ctd)))


#     #Get info
#     zooms = img_nib.header.get_zooms() #Voxel sizes
#     axs_code = nio.ornt2axcodes(nio.io_orientation(img_nib.affine)) #Image orientation
#     ctd_code = ctd_list[0] #Centroid orientation
#     data_shape_voxels = img_nib.header.get_data_shape() #Shape of data
#     data_shape_mm = np.array(data_shape_voxels)*np.array(zooms) #Data measures


#     if axs_code[0] in ['L','R']:
#         LR = 0
#     if axs_code[0] in ['A','P']:
#         AP = 0
#     if axs_code[0] in ['S','I']:
#         SI = 0

#     if axs_code[1] in ['L','R']:
#         LR = 1
#     if axs_code[1] in ['A','P']:
#         AP = 1
#     if axs_code[1] in ['S','I']:
#         SI = 1

#     if axs_code[2] in ['L','R']:
#         LR = 2
#     if axs_code[2] in ['A','P']:
#         AP = 2
#     if axs_code[2] in ['S','I']:
#         SI = 2
    
#     dim1_list_new = data_shape_mm[LR]
#     dim2_list_new = data_shape_mm[AP]
#     dim3_list_new = data_shape_mm[SI]
#     dim1_list.append(dim1_list_new)
#     dim2_list.append(dim2_list_new)
#     dim3_list.append(dim3_list_new)


#     if dim1_list_new > 512:
#         print(str(subject) + " is too big in dimension 1. Size is "+ str(dim1_list_new))
#         continue
#     if dim2_list_new > 512:
#         print(str(subject) + " is too big in dimension 2. Size is "+ str(dim2_list_new))
#         continue
#     if dim3_list_new > 1024:
#         print(str(subject) + " is too big in dimension 3. Size is "+ str(dim3_list_new))
#         continue

#     ctd_overview = []

#     for ctd in ctd_list[1:]:
#         if 17 <= ctd[0] <= 24:
#             ctd_overview.append(ctd[0])

#     if ctd_overview == []: #Hvis der ikke er nogen centroids i spændet.
#         print("No centroids in range..")
#         continue
#     else:
#         min_ctd = min(ctd_overview)
#         max_ctd = max(ctd_overview)
#         print("Yes, we have centroids. Range is: ["+str(min_ctd)+","+str(max_ctd)+"]")
#         if max_ctd >= 24 and min_ctd >= 18:
#             print("YESYEYSYESYEYYSEYYSYESEYSEYYSESEYEYSYEYEYESESYSYEYSYSEYSYEYSEYSEYE")
#         else:
#             continue
#             #print("Dropping it, because there are less than three centroids visible")
#             print("Dropping it, because not all centroids are visible")
            

#     compatible_subjects.append(subject)


# # with open(os.path.join(Output_folder,"list_of_subjects_"+data_type+"_VERSE"), "wb") as fp:   #Pickling
# #     pickle.dump(compatible_subjects, fp)

# # print(len(compatible_subjects))