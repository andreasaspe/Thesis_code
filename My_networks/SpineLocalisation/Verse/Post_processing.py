import os
from os import listdir
import numpy as np
import nibabel as nib
#from data_utilities import *
import torch
from my_data_utils import BoundingBox, RescaleBoundingBox
from my_plotting_functions import *
import pandas as pd
import pickle

#THIS SCRIPT FINDS BOUNDING BOX COORDINATES AND PLOT THESE

#######################################################
#################### CONTROL PANEL ####################
#######################################################
#Define scans
all_scans = 1 #Set to 1 if you want to preprocess all scans
list_of_subjects = ['sub-verse540'] #List of subjects #521? 574? verse563 er den med hovedet, verse509  - for stor box, hvis det er 0.3
#509 viser forskellen mellem om y_lower og y_upper skal være +-6 eller +-12. 
#gl428 er helt af helvedes til! Men det er fordi det er helt oppe i nakken!
#Går skidt:
#578
#563
#650 kan slet ikke køre

#Don't touch
New_orientation = ('L', 'A', 'S')
dim1_new = 64
dim2_new = 64
dim3_new = 128
New_voxel_size = 8 # [mm]

#Define directories
dir_heatmap = '/Users/andreasaspe/Documents/Data/Verse20/SpineLocalisation/Verse20_test_predictions' #'/scratch/s174197/data/Verse20/Verse20_test_predictions'  #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_test_predictions' #'/scratch/s174197/data/Verse20/Verse20_test_unpacked' #'/scratch/s174197/data/Verse20/Verse20_test_unpacked' #'/Users/andreasaspe/Documents/Data/Verse20_training' #r'C:\Users\PC\Documents\Andreas_s174197\dataset-verse20training_unpacked' #'/zhome/bb/f/127616/Documents/Thesis/Rawdata_training'
dir_image_original = '/Users/andreasaspe/Documents/Data/Verse20/SpineLocalisation/Verse20_test_reoriented/img' #'/scratch/s174197/data/Verse20/Verse20_test_unpacked' #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_test_unpacked'
padding_specifications_dir = '/Users/andreasaspe/Documents/Data/Verse20/SpineLocalisation/Padding_specifications/pad_test' #'/scratch/s174197/data/Verse20/Padding_specifications/pad_test.pkl'
#######################################################
#######################################################
#######################################################


#Define list of scans
if all_scans:
    all_subjects = []
    for filename in listdir(dir_image_original):
        subject = filename.split("_")[0]
        if subject.find('verse') != -1: #PLOTTER KUN VERSE. IKKE GL
            all_subjects.append(subject)
    all_subjects = np.unique(all_subjects)
else:
    all_subjects = list_of_subjects


x_list = []
y_list = []
z_list = []

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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #device = torch.device('cpu')
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
    filename_img = [f for f in listdir(dir_image_original) if (f.startswith(subject) and f.endswith('img.nii.gz'))][0]
    img_nib = nib.load(os.path.join(dir_image_original,filename_img))
    img_data = np.asanyarray(img_nib.dataobj, dtype=img_nib.dataobj.dtype)

    img_data[img_data<-200] = -200
    img_data[img_data>1000] = 1000

    #Normalise houndsfield units
    img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
    
    #Open heatmaps
    with open(padding_specifications_dir, 'rb') as f:
            padding_specifications = pickle.load(f) 
    restrictions = padding_specifications[subject]

    #Finding bounding box coordinates
    bb_coordinates, COM = BoundingBox(heatmap_data,restrictions)
    
    old_zooms = (8,8,8)
    new_zooms = img_nib.header.get_zooms()
    new_bb_coordinates, new_COM = RescaleBoundingBox(new_zooms,old_zooms,bb_coordinates,COM,restrictions)
    
    print(new_COM)

    x_min, x_max, y_min, y_max, z_min, z_max = bb_coordinates

    x_list.append((x_max-x_min)*8)
    y_list.append((y_max-y_min)*8)
    z_list.append((z_max-z_min)*8)

    
    #show_slices_dim1(img_data)
    show_boundingbox_dim1(img_data,new_bb_coordinates,subject,no_slices=40)
    # show_boundingbox_dim2(img_data,new_bb_coordinates,subject,no_slices=40)
    # show_boundingbox_dim3(img_data,new_bb_coordinates,subject,no_slices=100)

print()