#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 15:06:44 2023

@author: andreasaspe
"""


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
#from SpineLocalisationNet import *
#from Create_dataset import LoadData
from my_data_utils import *
import importlib

#Playing around

### ADJUST ###
heatmap_dir = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation2/Verse20_training_heatmaps_HUafter' #for_making_figure/output' #'/Users/andreasaspe/Documents/Data/Verse20/SpineLocalisation/Verse20_training_heatmaps' #Predictions or ground truth.
img_dir = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation2/Verse20_training_prep_HUafter/img' #'/Users/andreasaspe/Documents/Data/Verse20/SpineLocalisation/Verse20_training_prep/img' #Image folder of prepped data

all_scans = 1 #Set to 1 if you want to preprocess all scans
list_of_subjects = ['sub-verse833'] #['sub-verse503','sub-verse506','sub-verse525'] #524 går ud over! sub-verse570 sub-verse764 'sub-verse578' sub-verse563 og 60 List of subjects, 'sub-verse510'512 #verse526 var før og efter dårlig med bounding box
#Sketchy?
#Måske lidt mere på z-aksen?
#sub-verse809
#sub-verse764 - too low heatmap values.
#sub-verse710 ????
#gl195!!!

#Dårlig
#gl428
#gl279


#God
#verse517

##############

#Define list of scans
if all_scans:
    all_subjects = []
    for filename in listdir(heatmap_dir): #img_dir
        subject = filename.split("_")[0]
        if subject.find('verse') != -1: #PLOTTER KUN VERSE. IKKE GL
            all_subjects.append(subject)
    all_subjects = np.unique(all_subjects)
else:
    all_subjects = list_of_subjects

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #device = torch.device('cpu')

for subject in all_subjects:
    filename_heatmap = [f for f in listdir(heatmap_dir) if f.startswith(subject)][0]
    heatmap_file_dir = os.path.join(heatmap_dir, filename_heatmap)
    try: #Load tensor or numpy
        heatmap_data = torch.load(heatmap_file_dir, map_location=device)
        heatmap_data = heatmap_data.detach().numpy()
        #Normalize
        #heatmap_data = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())
    except:
        heatmap_nib = nib.load(heatmap_file_dir)
        heatmap_data = np.asanyarray(heatmap_nib.dataobj, dtype=np.float32)
        #Normalize
        #heatmap_data = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())

    
    filename_img = subject + "_img.nii.gz"
    img_nib = nib.load(os.path.join(img_dir,filename_img))
    img_data = np.asanyarray(img_nib.dataobj, dtype=np.float32)
    
    # show_slices_dim1(img_data,subject)
    show_heatmap_img_dim1(img_data,heatmap_data,subject)
    # show_heatmap_dim1(heatmap_data,subject)
    
    # no_slices=50
    # plt.style.use('default')
    # dim = heatmap_data.shape[0]
    # slice_step = int(dim/no_slices)
    # if slice_step == 0:
    #     slice_step = 1
    # for i in range(0,dim,slice_step):
    #     fig, ax = plt.subplots()
    #     heatmap_plot = ax.imshow(heatmap_data[i,:,:].T, cmap='hot',origin="lower", vmin=0, vmax=1) #Jo større tal ved alpha, jo bredere kurve  ax.set_title('dim, '+str(subject)+", Slice: "+str(i))
    #     plt.colorbar(heatmap_plot)
    #     plt.axis('off')
    #     ax.set_title("Dim1 "+subject+", Slice: "+str(i))
    #     # if i == 50:
    #     #     #figsize 8,8
    #     #     plt.tight_layout()
    #     #     plt.axis('off')
    #     #     plt.savefig('Heatmap', transparent=True))
    #     plt.show()
    
    
    #show_slices_dim1(img_data, subject)
    
    

    # bb_coordinates = BoundingBox(heatmap_data)

    #show_slices_dim2(img_data,no_slices=40)
    # show_boundingbox_dim1(img_data,bb_coordinates,subject)
    # show_boundingbox_dim2(img_data,heatmap_data,subject)
    # show_boundingbox_dim3(img_data,heatmap_data,subject)
    #show_heatmap_dim1(img_data,heatmap_data,subject,alpha=0.5,no_slices=40)
    # show_heatmap_dim2(img_data,heatmap_data,subject,alpha=0.4,no_slices=40)
    # show_heatmap_dim3(img_data,heatmap_data,subject,alpha=0.4,no_slices=60)