#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 12:24:06 2023

@author: andreasaspe
"""

import os
from os import listdir
import numpy as np
from tqdm import tqdm
import nibabel as nib
import pandas as pd
import cv2
from my_plotting_functions import *
from my_data_utils import dice_score
from my_data_utils import *
from my_plotting_functions import *


#######################################################
#################### CONTROL PANEL ####################
#######################################################
#Define scans
all_scans = 1
list_of_subjects = ['sub-verse558']

data_type = 'test'

#HUSK AT GENNEMGÅ FIL FOR FLERE TING AT ÆNDRE
#Define directories Titans
dir_segmentations = '/scratch/s174197/data/Verse20/Predictions_from_titans/FULL_SEGMENTATIONS_batchnorm_beforeCCA_evenbetterrotation'
dir_GT = '/scratch/s174197/data/Verse20/Predictions_from_titans/FULL_SEGMENTATIONS_batchnorm_GT'
dir_data_stage3 = '/scratch/s174197/data/Verse20/VertebraeSegmentation/Verse20_'+data_type+'_prep_NOPADDING'
predictions_dataframe_folder = '/scratch/s174197/data/Verse20/Predictions_dataframes_from_titans'

#Define directories MAC
# dir_segmentations = '/Users/andreasaspe/Documents/Data/Verse20/Predictions_segmentations'
# dir_data_stage3 = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeSegmentation/Verse20_'+data_type+'_prep_NOPADDING' #'/zhome/bb/f/127616/Documents/Thesis/Rawdata_training'
# predictions_dataframe_folder = '/Users/andreasaspe/Documents/Data/Verse20/Predictions_dataframe'

#######################################################
#######################################################
#######################################################

#Convert directories
dir_segmentations = os.path.join(dir_segmentations,data_type)
dir_GT = os.path.join(dir_GT,data_type)
dir_stage3_img = os.path.join(dir_data_stage3,'img')
dir_predictions_dataframe_folder = os.path.join(predictions_dataframe_folder,data_type,'evenbetterrotation_before')

if not os.path.exists(dir_predictions_dataframe_folder):
    os.makedirs(dir_predictions_dataframe_folder)


#Define list of scans
if all_scans:
    all_subjects = []
    for filename in listdir(dir_GT):
        subject = filename.split("_")[0]
        #if subject.find('verse') != -1: #PLOTTER KUN VERSE. IKKE GL
        all_subjects.append(subject)
    all_subjects = np.unique(all_subjects)
    #Sorterer fil '.DS' fra
    all_subjects = all_subjects[all_subjects != '.DS']
    all_subjects = all_subjects.tolist()
else:
    all_subjects = list_of_subjects



DSC_list = []
avg_DSC_list = []
Hausdorff_list = []
avg_Hausdorff_list = []



for counter, subject in enumerate(tqdm(all_subjects)):
    
    print()
    print()
    print(subject)
    print()
    print()
    
    #Define filenames
    # FOR MAC
    # filename_msk_GT = [f for f in listdir(dir_segmentations) if (f.startswith(subject) and f.endswith('GT.nii.gz'))][0]
    # filename_msk_pred = [f for f in listdir(dir_segmentations) if (f.startswith(subject) and f.endswith('PREDICTION.nii.gz'))][0]
    # filename_img = [f for f in listdir(dir_stage3_img) if (f.startswith(subject) and f.endswith('img.nii.gz'))][0]

    #TITANS
    filename_msk_GT = [f for f in listdir(dir_GT) if (f.startswith(subject) and f.endswith('GT.nii.gz'))][0]
    filename_msk_pred = [f for f in listdir(dir_segmentations) if (f.startswith(subject) and f.endswith('PREDICTIONbefore.nii.gz'))][0]
    filename_img = [f for f in listdir(dir_stage3_img) if (f.startswith(subject) and f.endswith('img.nii.gz'))][0]

    #Load Nifti files
    msk_nib_GT = nib.load(os.path.join(dir_GT,filename_msk_GT))
    msk_nib_pred = nib.load(os.path.join(dir_segmentations,filename_msk_pred))
    img_nib = nib.load(os.path.join(dir_stage3_img,filename_img))

    #Get data
    data_msk_GT = np.asanyarray(msk_nib_GT.dataobj, dtype=np.float32)
    data_msk_pred = np.asanyarray(msk_nib_pred.dataobj, dtype=np.float32)
    data_img = np.asanyarray(img_nib.dataobj, dtype=np.float32)
    
    
    v_numbers = np.unique(data_msk_pred)

    DSC_all = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]
    Hausdorff_all = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]

    
    for i in range(len(v_numbers)-1): #Fordi vi gider ikke have 0 med
        v_number = int(v_numbers[i+1]) #i+1 fordi vi ikke gider at have 0 med
        msk_GT = np.where(data_msk_GT == v_number,1,0)
        msk_pred = np.where(data_msk_pred == v_number,1,0)
        
        
        #Get indices to crop it tigthly
        x_indices_GT, y_indices_GT, z_indices_GT = np.where(msk_GT != 0)
        x_indices_pred, y_indices_pred, z_indices_pred = np.where(msk_pred != 0)
        
        # Calculate the bounding box coordinates
        x_min = np.min(np.concatenate((x_indices_GT,x_indices_pred)))
        x_max = np.max(np.concatenate((x_indices_GT,x_indices_pred)))
        y_min = np.min(np.concatenate((y_indices_GT,y_indices_pred)))
        y_max = np.max(np.concatenate((y_indices_GT,y_indices_pred)))
        z_min = np.min(np.concatenate((z_indices_GT,z_indices_pred)))
        z_max = np.max(np.concatenate((z_indices_GT,z_indices_pred)))
        
        #Crop
        msk_GT = msk_GT[x_min:x_max+1,y_min:y_max+1,z_min:z_max+1]
        msk_pred = msk_pred[x_min:x_max+1,y_min:y_max+1,z_min:z_max+1]
        
        #Find contours!
        
        contour_image_GT = find_3d_contours(msk_GT)
        contour_image_pred = find_3d_contours(msk_pred)
        
        DSC = dice_score(msk_GT,msk_pred)
        print(number_to_name[v_number])
        print("Dice is {:.2f}".format(DSC))
        Hausdorff = hausdorff_distance_3d(contour_image_GT,contour_image_pred)
        print("Hausdorff dist is {:.2f}".format(Hausdorff))
        
        
        DSC_all[abs(v_number-24)] = DSC #L5 er første plads i stedet for sidste
        Hausdorff_all[abs(v_number-24)] = Hausdorff
        
    print()
    
    DSC_list.append(DSC_all)
    avg_DSC_list.append(np.nanmean(DSC_all))
    Hausdorff_list.append(Hausdorff_all)
    avg_Hausdorff_list.append(np.nanmean(Hausdorff_all))

column_names = ['L5','L4','L3','L2','L1','T12','T11','T10']

DSC_dataframe = pd.DataFrame(DSC_list, columns=column_names)
DSC_dataframe.insert(0, 'Average distance', avg_DSC_list)
DSC_dataframe.insert(0, 'subjects', all_subjects)
DSC_dataframe.to_csv(os.path.join(dir_predictions_dataframe_folder,'df_DSC.csv'), index=False)


HAUSDORFF_dataframe = pd.DataFrame(Hausdorff_list, columns=column_names)
HAUSDORFF_dataframe.insert(0, 'Average distance', avg_Hausdorff_list)
HAUSDORFF_dataframe.insert(0, 'subjects', all_subjects)
HAUSDORFF_dataframe.to_csv(os.path.join(dir_predictions_dataframe_folder,'df_HAUSDORFF.csv'), index=False)
 
          

    