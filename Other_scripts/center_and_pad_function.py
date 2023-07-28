#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 15:51:26 2023

@author: andreasaspe
"""

import numpy as np
from my_plotting_functions import *
from copy import deepcopy


def center_and_pad(data,new_dim,pad_value,centroid=None):
    dim1, dim2, dim3 = data.shape
    dim1_new, dim2_new, dim3_new = output_dim
    
    if centoid != None:
        x_start = max(x-int(dim1_new/2),0) #64
        x_end = min(x+int(dim1_new/2)-1,dim1-1) #63
        y_start = max(y-int(dim2_new/2),0)
        y_end = min(y+int(dim2_new/2)-1,dim2-1)
        z_start = max(z-int(dim3_new/2),0)
        z_end = min(z+int(dim3_new/2)-1,dim3-1)
        
        data_cropped = deepcopy(data_img[x_start:x_end+1,y_start:y_end+1,z_start:z_end+1])
    else:
        data_cropped = deepcopy(data_img)
    
    #Get dimensions after cropping
    dim1, dim2, dim3 = data_new.shape
    
    #Calculate padding in each side (volume should be centered)
    padding_dim1 = (dim1_new-dim1)/2
    padding_dim2 = (dim2_new-dim2)/2
    padding_dim3 = (dim3_new-dim3)/2
    
    #Calculate padding in each side by taking decimal values into account
    #Dim1
    if padding_dim1 > 0:
        if np.floor(padding_dim1) == padding_dim1:
            pad1 = (int(padding_dim1),int(padding_dim1))
        else:
            pad1 = (int(np.floor(padding_dim1)),int(np.floor(padding_dim1)+1))
    else:
        pad1 = (0,0)
    #Dim2
    if padding_dim2 > 0:
        if np.floor(padding_dim2) == padding_dim2:
            pad2 = (int(padding_dim2),int(padding_dim2))
        else:
            pad2 = (int(np.floor(padding_dim2)),int(np.floor(padding_dim2)+1))
    else:
        pad2 = (0,0)
    #Dim3
    if padding_dim3 > 0:
        if np.floor(padding_dim3) == padding_dim3:
            pad3 = (int(padding_dim3),int(padding_dim3))
        else:
            pad3 = (int(np.floor(padding_dim3)),int(np.floor(padding_dim3)+1))
    else:
        pad3 = (0,0)

    #Doing padding
    data_cropped=np.pad(data_cropped, (pad1, pad2, pad3), constant_values = pad_value)
    
    return data_cropped
    
    

data = np.ones((29,128,96)) #(29,128,96)
centroid = [20,20,20]
output_dim = (128,128,96)

cropped_data = center_and_pad(data,centroid,output_dim)

show_slices_dim1(cropped_data,'dontknow')
