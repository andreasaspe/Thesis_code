# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 21:39:53 2023

@author: PC
"""

import os

folder_path = r'E:/s174197/data_RH/Predictions_newaffine/Segmentations_afterCCA'  # Replace this with the actual path to your folder

for filename in os.listdir(folder_path):
    if filename.endswith('.nii.gz') and '_PREDICTION' in filename:
        # Split the filename into parts using underscores as separators
        parts = filename.split('_')
        # Extract the PREDICTIONafter parts
        PREDICTIONafter = parts[4].split('.')[0]  # Remove the .nii extension from the last part
        # Concatenate the parts with a hyphen and add back the .nii extension
        new_filename = f"{parts[0]}_{parts[1]}_{parts[2]}_{parts[3]}-PREDICTIONafter.nii.gz"
        
        # Get the full path of the file
        old_filepath = os.path.join(folder_path, filename)
        new_filepath = os.path.join(folder_path, new_filename)
        
        # Rename the file
        os.rename(old_filepath, new_filepath)