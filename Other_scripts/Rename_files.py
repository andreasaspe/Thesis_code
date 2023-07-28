#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 12:50:55 2023

@author: s174197
"""

from os import listdir
import os
import shutil

#PUT IN ENDING
dir_data = '/zhome/bb/f/127616/Documents/Thesis/Outputs_new'

all_files = [f for f in listdir(dir_data) if not f.startswith('.')] #Remove file .DS_Store - could also be: if not f.startswith('.')

ending = ".json" #'ct.nii.gz'
new_ending = "_ctd.json" #'img.nii.gz'

for filename in all_files:
    if filename.endswith(ending):
        name = filename[:-len(ending)]
        new_filename = name + new_ending
        source_dir = os.path.join(dir_data,filename)
        destination_dir = os.path.join(dir_data,new_filename)
        shutil.move(source_dir, destination_dir)
        print("Name changed from: "+str(filename))
        print("Name changed to: "+str(new_filename))
        print()
