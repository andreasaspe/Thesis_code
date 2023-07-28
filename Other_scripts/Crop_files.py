#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 10:10:39 2023

@author: andreasaspe
"""

import os
from PIL import Image
import matplotlib.pyplot as plt

# Folder path containing the images to crop
folder_path = '/Users/andreasaspe/Library/Mobile Documents/com~apple~CloudDocs/DTU/12.semester/Thesis/Figures/RH_visualresults/Vertebraelocalisation_anomalies/non-cropped' #'/Users/andreasaspe/Library/Mobile Documents/com~apple~CloudDocs/DTU/12.semester/Thesis/Figures/VertebraeLocalisation/VertebraeLocalisation2_results/non-cropped' #r'E:/s174197/Thesis/Figures/Fractures/Method/non-cropped' #'/Users/andreasaspe/Library/Mobile Documents/com~apple~CloudDocs/DTU/12.semester/Thesis/Figures/VertebraeLocalisation/VertebraeLocalisation2_results/non-cropped' #'/Users/andreasaspe/Library/Mobile Documents/com~apple~CloudDocs/DTU/12.semester/Thesis/Figures/SpineLocalisation/Verse/Predictions_to_report/non-cropped' #"/Users/andreasaspe/Library/Mobile Documents/com~apple~CloudDocs/DTU/12.semester/Thesis/Figures/VertebraeLocalisation/VertebraeLocalisation2_results/non-cropped" #"/Users/andreasaspe/Library/Mobile Documents/com~apple~CloudDocs/DTU/12.semester/Thesis/Figures/SpineLocalisation/Verse/Predictions_to_report/non-cropped" #"/Users/andreasaspe/Library/Mobile Documents/com~apple~CloudDocs/DTU/12.semester/Thesis/Figures/VertebraeLocalisation/VertebraeLocalisation2_results/non-cropped" #"/Users/andreasaspe/iCloud/DTU/12.semester/Thesis/Figures/SpineLocalisation/Verse/BoundingBox/non-cropped"

# # Create a subfolder named "cropped" if it doesn't exist
cropped_folder = os.path.join(os.path.dirname(folder_path), "cropped") #os.path.dirname er parent folder
os.makedirs(cropped_folder, exist_ok=True)




# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path) and filename != '.DS_Store':
        # Open the image file
        image = Image.open(file_path)
        
        # image.show()
        # break
        
        #Original
        width, height = image.size   # Get dimensions
        left = 11 #width/4
        top = 34 #height/4
        right = width-11 #3 * width/4
        bottom = height-11 #3 * height/4
        
        #Other
        # width, height = image.size   # Get dimensions
        # left = 10 #width/4
        # top = 0 #height/4
        # right = width #3 * width/4
        # bottom = height-10 #3 * height/4
        
        #Crop the image
        cropped_image = image.crop((left, top, right, bottom))
        
        # cropped_image.show()
        
        # break
        
        # Save the cropped image in the "cropped" subfolder with the same filename
        cropped_path = os.path.join(cropped_folder, filename)
        cropped_image.save(cropped_path)

        # Close the image file
        image.close()

print("Cropping and displaying completed.")




#Original
        # width, height = image.size   # Get dimensions
        # left = 11 #width/4
        # top = 34 #height/4
        # right = width-11 #3 * width/4
        # bottom = height-11 #3 * height/4