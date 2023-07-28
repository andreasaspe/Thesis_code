import os
from os import listdir
import numpy as np
import nibabel as nib
#from data_utilities import *
import torch
import sys
sys.path.append('E:/Andreas_s174197/Thesis/MY_CODE/utils')
from my_data_utils import BoundingBox, RescaleBoundingBox
from my_plotting_functions import *
import pandas as pd
import pickle

#THIS SCRIPT FINDS BOUNDING BOX COORDINATES AND PLOT THESE

#######################################################
#################### CONTROL PANEL ####################
#######################################################
#Define scans
all_scans = 0 #Set to 1 if you want to preprocess all scans
list_of_subjects = ['VERTEBRAE_HEALTHY_0001_SERIES0010'] #List of subjects
# with open("E:\Andreas_s174197\Thesis\My_code\Other_scripts\list_of_subjects", "rb") as fp:   # Unpickling
#     list_of_subjects = pickle.load(fp)

#Don't touch
New_orientation = ('L', 'A', 'S')
dim1_new = 64
dim2_new = 64
dim3_new = 128
New_voxel_size = 8 # [mm]

#Define directories
dir_heatmap = r'E:\Andreas_s174197\data_RH\heatmap_predictions_temp' #'/scratch/s174197/data/Verse20/Verse20_test_predictions'  #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_test_predictions' #'/scratch/s174197/data/Verse20/Verse20_test_unpacked' #'/scratch/s174197/data/Verse20/Verse20_test_unpacked' #'/Users/andreasaspe/Documents/Data/Verse20_training' #r'C:\Users\PC\Documents\Andreas_s174197\dataset-verse20training_unpacked' #'/zhome/bb/f/127616/Documents/Thesis/Rawdata_training'
dir_image_original = r'F:\DTU-Vertebra-1\NIFTI' #'/scratch/s174197/data/Verse20/Verse20_test_unpacked' #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_test_unpacked'
padding_specifications_dir = 'E:/Andreas_s174197/data_RH/Padding_specifications/pad_temp' #'/Users/andreasaspe/Documents/Data/Verse20/Padding_specifications/pad_test' #'/scratch/s174197/data/Verse20/Padding_specifications/pad_test.pkl'
#######################################################
#######################################################
#######################################################


#Define list of scans
if all_scans:
    all_subjects = []
    for filename in listdir(dir_image_original):
        subject = filename.split("_")[0]
        #if subject.find('verse') != -1: #PLOTTER KUN VERSE. IKKE GL
        all_subjects.append(subject)
    all_subjects = np.unique(all_subjects)
else:
    all_subjects = list_of_subjects



for subject in all_subjects:
    print("\n\n")
    print("       SUBJECT: "+str(subject)+"\n")
    #Load heatmaps
    try: #Prøver. Hvis ikke filen findes, så er det fordi jeg frasortede den på et tidspunkt.
        filename_heatmap = [f for f in listdir(dir_heatmap) if f.startswith(subject)][0]
        heatmap_file_dir = os.path.join(dir_heatmap, filename_heatmap)
    except:
        continue
    try: #Load tensor or numpy
        device = torch.device('cpu') #device = torch.device('cpu')
        heatmap_data = torch.load(heatmap_file_dir, map_location=device)
        heatmap_data = heatmap_data.detach().numpy()
        #Normalize
        heatmap_data = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())
    except:
        heatmap_nib = nib.load(heatmap_file_dir)
        heatmap_data = np.asanyarray(heatmap_nib.dataobj, dtype=np.float32)
        #Normalize
        heatmap_data = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())
        
    # load image
    img_nib = nib.load(os.path.join(dir_image_original,subject+'.nii.gz'))
    img_data = np.asanyarray(img_nib.dataobj, dtype=img_nib.dataobj.dtype)

    #Open heatmaps
    with open(padding_specifications_dir, 'rb') as f:
            padding_specifications = pickle.load(f) 
    restrictions = padding_specifications[subject]

    #Finding bounding box coordinates
    bb_coordinates = BoundingBox(heatmap_data,restrictions)
    
    old_zooms = (8,8,8)
    new_zooms = img_nib.header.get_zooms()
    new_bb_coordinates = RescaleBoundingBox(new_zooms,old_zooms,bb_coordinates,restrictions)
    
    #show_slices_dim1(img_data)
    show_boundingbox_dim1(img_data,new_bb_coordinates,subject,no_slices=15)
    show_boundingbox_dim2(img_data,new_bb_coordinates,subject,no_slices=15)
    show_boundingbox_dim3(img_data,new_bb_coordinates,subject,no_slices=15)

