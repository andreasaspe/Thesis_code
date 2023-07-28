#General imports
import os
import sys
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
import nibabel as nib
#My own documents
from my_plotting_functions import *
from SpineLocalisationNet import *
from my_data_utils import *
import pickle

######################### CONTROL PANEL #########################
plot_loss = 0
plot_heatmaps = 1
#################################################################


#Plot loss
if plot_loss:
    ### ADJUST ###
    checkpoint_filename = 'checkpoint_batchsize1_lr0.0001_wd0.0005.pth'
    #checkpoint_filename = 'checkpoint_batchsize1_learningrate0.0001.pth'
    ##############

    #Define directories
    mac = '/Users/andreasaspe/Library/Mobile Documents/com~apple~CloudDocs/DTU/12.semester/Thesis/My_code/My_networks/Spine_Localisation/Checkpoints'
    GPUcluster = '/home/s174197/Thesis/My_code/My_networks/Spine_Localisation/Checkpoints'
    GPU1 = 'C:/Users/PC/Documents/Andreas_s174197/Thesis/My_code/My_networks/Spine_Localisation/Checkpoints'
    GPU2 = ''
    checkpoint_dir = os.path.join(GPU1,checkpoint_filename)
    
    #Call function
    Plot_loss(checkpoint_dir)

#Plot predictions
if plot_heatmaps:
    ### ADJUST ###
    heatmap_dir = r"E:\Andreas_s174197\data_RH\heatmap_predictions_temp" #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_test_heatmaps_predictions' #Predictions or ground truth.
    img_dir = r"E:\Andreas_s174197\data_RH\data_prep_temp\img" #Users/andreasaspe/Documents/Data/Verse20/Verse20_test_prep/img' #Image folder of prepped data
    padding_specifications_dir = 'E:/Andreas_s174197/data_RH/Padding_specifications/pad_temp'

    all_scans = 0 #Set to 1 if you want to preprocess all scans
    list_of_subjects = ['VERTEBRAE_HEALTHY_0001_SERIES0010'] #List of subjects
    # with open("E:\Andreas_s174197\Thesis\My_code\Other_scripts\list_of_subjects", "rb") as fp:   # Unpickling
    #     list_of_subjects = pickle.load(fp)
    #Dårlig
    #gl428
    #VERTEBRAE_FRACTURE_0200_SERIES0005
    
    
    #God
    #verse517

    ##############

    #Define list of scans
    if all_scans:
        all_subjects = []
        for filename in listdir(img_dir):
            subject = filename.split(".")[0]
            subjectname_without_img = subject[:-4]
            all_subjects.append(subjectname_without_img)
        all_subjects = np.unique(all_subjects)
    else:
        all_subjects = list_of_subjects


    device = torch.device('cpu') #device = torch.device('cpu')

    for subject in all_subjects:
        print(subject)
        print("\n\n\n")
        try: #Se om det er en af dem jeg har preprocessed. Ellers gå videre
            filename_heatmap = [f for f in listdir(heatmap_dir) if f.startswith(subject)][0]
            heatmap_file_dir = os.path.join(heatmap_dir, filename_heatmap)
        except:
            continue
        try: #Load tensor or numpy
            heatmap_data = torch.load(heatmap_file_dir, map_location=device)
            heatmap_data = heatmap_data.detach().numpy()
            #Normalize
            heatmap_data = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())
        except:
            heatmap_nib = nib.load(heatmap_file_dir)
            heatmap_data = np.asanyarray(heatmap_nib.dataobj, dtype=np.float32)
            #Normalize
            heatmap_data = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())

        
        filename_img = subject + "_img.nii.gz"
        img_nib = nib.load(os.path.join(img_dir,filename_img))
        img_data = np.asanyarray(img_nib.dataobj, dtype=np.float32)
        
        with open(padding_specifications_dir, 'rb') as f:
            padding_specifications = pickle.load(f)
            
        try: #Måske er det ikke en af dem, jeg har lavet padding på?
            restrictions = padding_specifications[subject]
        except:
            continue
        
        bb_coordinates = BoundingBox(heatmap_data,restrictions)

        #show_slices_dim1(img_data,no_slices=40)
        show_boundingbox_dim1(img_data,bb_coordinates,subject)
        # show_boundingbox_dim2(img_data,bb_coordinates,subject)
        # show_boundingbox_dim3(img_data,bb_coordinates,subject)
        #show_heatmap_dim1(img_data,heatmap_data,subject,alpha=0.4,no_slices=40)
        #show_heatmap_dim2(img_data,heatmap_data,subject,alpha=0.4,no_slices=40)
        #show_heatmap_dim3(img_data,heatmap_data,subject,alpha=0.4,no_slices=60)
