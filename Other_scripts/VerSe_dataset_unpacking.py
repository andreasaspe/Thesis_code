#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 12:04:46 2023

@author: s174197
"""

from os import listdir
import os
import shutil

# parent_directory = "/zhome/bb/f/127616/Documents/Thesis/dataset-01training/"

# all_folders = [f for f in listdir(parent_directory) if not f.startswith('.')] #Remove file .DS_Store

# for folder in all_folders:
#     all_subfolders = [f for f in listdir(os.path.join(parent_directory,folder)) if not f.startswith('.')] #Remove file .DS_Store
#     for subfolder in all_subfolders:
#         if folder == 'rawdata':
#             filename_img = [f for f in listdir(os.path.join(folder,subfolder)) if f.find('')][0]
        
        
#         fullstring.find(substring) != -1:
    
#Input directories
dir_rawdata = '/scratch/s174197/data/Verse19/dataset-verse19validation/rawdata' #'/Users/andreasaspe/Documents/Data/dataset-verse20test/rawdata' #r'C:\Users\PC\Documents\Andreas_s174197\dataset-verse20training\dataset-01training\rawdata' #'/zhome/bb/f/127616/Documents/Thesis/dataset-01training/rawdata' #'/Users/andreasaspe/Documents/Data/dataset-01training/rawdata'
dir_derivatives = '/scratch/s174197/data/Verse19/dataset-verse19validation/derivatives' #'/Users/andreasaspe/Documents/Data/dataset-verse20test/derivatives' #r'C:\Users\PC\Documents\Andreas_s174197\dataset-verse20training\dataset-01training\derivatives' #'/zhome/bb/f/127616/Documents/Thesis/dataset-01training/derivatives/' #'/Users/andreasaspe/Documents/Data/dataset-01training/derivatives'
#Output directories (Will create a folder if it does not exist)
dir_destination = '/scratch/s174197/data/Verse19/Verse19_validation_unpacked' #'/Users/andreasaspe/Documents/Data/Verse20_test' #r'C:\Users\PC\Documents\Andreas_s174197\dataset-verse20training_unpacked' #'/zhome/bb/f/127616/Documents/Thesis/Rawdata_training'

#Define list of scans
scans = [f for f in listdir(dir_rawdata) if f.startswith('sub')] #Remove file .DS_Store
#Create training folder if it does not exist
if not os.path.exists(dir_destination):
   os.makedirs(dir_destination)

#FOR LOOP START
for subject in scans:
    print("       SUBJECT: "+str(subject)+"\n")
    try:
        # Define file names
        filename_img = [f for f in listdir(os.path.join(dir_rawdata,subject)) if f.endswith('.gz')][0]
        filename_msk = [f for f in listdir(os.path.join(dir_derivatives,subject)) if f.endswith('.gz')][0]
        filename_ctd = [f for f in listdir(os.path.join(dir_derivatives,subject)) if f.endswith('subreg_ctd.json')][0]
        #Get directory of source
        source_img = os.path.join(dir_rawdata,subject,filename_img)
        source_msk = os.path.join(dir_derivatives,subject,filename_msk)
        source_ctd = os.path.join(dir_derivatives,subject,filename_ctd)
        #Get new file names for image
        ending = 'ct.nii.gz'
        new_ending = 'img.nii.gz'
        name = filename_img[:-len(ending)]
        new_filename_img = name + new_ending
        destination_dir_img = os.path.join(dir_destination,new_filename_img)
        #Get final directory of destination
        dir_destination_img = os.path.join(dir_destination,new_filename_img)
        dir_destination_msk = os.path.join(dir_destination,filename_msk)
        dir_destination_ctd = os.path.join(dir_destination,filename_ctd)
        #Move files
        shutil.move(source_img, dir_destination_img)
        shutil.move(source_msk, dir_destination_msk)
        shutil.move(source_ctd, dir_destination_ctd)
        print("Subject "+str(subject)+" has been moved.")
    except:
        print("Subject "+str(subject)+" has already been moved.")