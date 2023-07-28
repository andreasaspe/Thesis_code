#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 20 13:49:58 2023

@author: andreasaspe
"""

#Windows shortcuts
#Comment: Ctrl+k+c (c for comment)
#Uncomment: Ctrl+k+u (u for uncomment)

import os
import numpy as np
from torch.utils.data import Dataset
import nibabel as nib
import torch
import torchvision.transforms.functional as F
from random import *
import SimpleITK as sitk
#My moduiles
from data_utilities import load_centroids
from my_data_utils import gaussian_kernel_3d
from volumentations import *
from scipy.ndimage import rotate

    

class LoadData2(Dataset):
    def __init__(self, img_dir, heatmap_dir, transform=None):
        self.img_dir = img_dir
        self.heatmap_dir = heatmap_dir
        self.images = os.listdir(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir,self.images[index])
        heatmap_path = os.path.join(self.heatmap_dir,self.images[index].replace('img.nii.gz','heatmap.nii.gz'))


        # Assuming you have a 3D image loaded with SimpleITK
        image = sitk.ReadImage(img_path)  # Replace with the path to your image
        
        # Convert the image to a numpy array
        image_array = sitk.GetArrayFromImage(image)
        
        
        dim1, dim2, dim3 = image_array
        
        # Perform the rotation
        rotation_angle_degrees = 30.0
        rotation_axis = [1, 0, 0]  # Rotation axis (X-axis in this example)
        
        # Create the VersorTransform
        rotation_center = np.array(image_array.shape) / 2  # Rotation center at the image center
        rotation_angle_radians = np.deg2rad(rotation_angle_degrees)
        rotation = sitk.VersorTransform(rotation_axis, rotation_angle_radians, rotation_center)
        
        # Apply the rotation to the image
        img = sitk.GetArrayFromImage(sitk.Resample(image, rotation))
        
        
        
        
        
        # Assuming you have a 3D image loaded with SimpleITK
        image = sitk.ReadImage(heatmap_path)  # Replace with the path to your image
        
        
        heatmaps = np.zeros((8,dim1,dim2,dim3))
            
        for i in range(8):
    
            # Apply the rotation to the image
            heatmap1 = sitk.GetArrayFromImage(sitk.Resample(image, rotation))


        # #Get data shape
        # dim1, dim2, dim3 =  img.header.get_data_shape()
        # #Convert to right format
        # img = np.asanyarray(img.dataobj, dtype=np.float32)
        # heatmap = np.asanyarray(heatmap.dataobj, dtype=np.float32)

        # #Crop images randomly if z-axis is too long
        # if dim3 > 128:
        #     #128-1 er nederste top-voxel, der kan vælges, hvor det croppede stadig kan være i. 128 er lige akkurat for meget (derfor vil jeg gerne have mulighed for at vælge 128 - fordi x[0:128] vil tage x fra 0 til 127)
        #     #Tilsvarende med dim3-1. Derfor vil jeg gerne have mulighed for at vælge dim3. Men for at kunne vælge dim3, så skal jeg sige range(128,dim3+1). Fordi den vælger altid én under max, dvs. dim3 er max. Prøv selv at teste ved at sige max(0,165).
        #     #Det fungerer sådan, fordi python er 0-indexeret.
        #     #Derfor siger jeg range(128,dim3).
        #     #Dvs. tag et tilfældigt valg imellem de to.
        #     top_voxel = np.random.choice(range(127,dim3)) #Random voxel. Det er den øverste værdi, en voxel kan have.
        #     #Crop image
        #     bottom_voxel = top_voxel-127 #Det er den ægte værdi voxel kan være
        #     img = img[:,:,bottom_voxel:top_voxel+1]
        #     heatmap = heatmap[:,:,:,bottom_voxel:top_voxel+1]


        #SUBJECT
        subject = self.images[index].split("_")[0]

        if self.transform is not None:
            aug = Compose([
                Rotate((-90, 90), (-90, 90), (0, 0), p=1.0),],
                p=1.0)
            data = {'image': img, 'mask': heatmap[0,:,:,:]}
            aug_data = aug(**data)
            img, heatmap = aug_data['image'], aug_data['mask']
            
            # random_rotation = randint(-15,-15)
            # img = F.rotate(angle = random_rotation,img=img,fill=-1)
            # heatmap = F.rotate(angle = random_rotation,img=heatmap,fill=0)
            
        #Convert to tensor
        # img = torch.from_numpy(img)
        # heatmap = torch.from_numpy(heatmap)

        # #Add one dimension to image to set #channels = 1. The heatmap is already in 4 dimensions with 8 channels from heatmap_generation script
        # img = img.unsqueeze(0) 
            
        return img, heatmap, subject
    
    
    

class LoadData(Dataset):
    
    
    
    def __init__(self, img_dir, heatmap_dir, transform=None):
        self.img_dir = img_dir
        self.heatmap_dir = heatmap_dir
        self.images = os.listdir(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir,self.images[index])
        heatmap_path = os.path.join(self.heatmap_dir,self.images[index].replace('img.nii.gz','heatmap.nii.gz'))

        #Load image and heatmap
        img = nib.load(img_path)
        heatmap = nib.load(heatmap_path)
        #Get data shape
        dim1, dim2, dim3 =  img.header.get_data_shape()
        #Convert to right format
        img = np.asanyarray(img.dataobj, dtype=np.float32)
        heatmap = np.asanyarray(heatmap.dataobj, dtype=np.float32)

        #Crop images randomly if z-axis is too long
        if dim3 > 128:
            #128-1 er nederste top-voxel, der kan vælges, hvor det croppede stadig kan være i. 128 er lige akkurat for meget (derfor vil jeg gerne have mulighed for at vælge 128 - fordi x[0:128] vil tage x fra 0 til 127)
            #Tilsvarende med dim3-1. Derfor vil jeg gerne have mulighed for at vælge dim3. Men for at kunne vælge dim3, så skal jeg sige range(128,dim3+1). Fordi den vælger altid én under max, dvs. dim3 er max. Prøv selv at teste ved at sige max(0,165).
            #Det fungerer sådan, fordi python er 0-indexeret.
            #Derfor siger jeg range(128,dim3).
            #Dvs. tag et tilfældigt valg imellem de to.
            top_voxel = np.random.choice(range(127,dim3)) #Random voxel. Det er den øverste værdi, en voxel kan have.
            #Crop image
            bottom_voxel = top_voxel-127 #Det er den ægte værdi voxel kan være
            img = img[:,:,bottom_voxel:top_voxel+1]
            heatmap = heatmap[:,:,:,bottom_voxel:top_voxel+1]


        #SUBJECT
        subject = self.images[index].split("_")[0]

        if self.transform is not None:
            # aug = Compose([
            #     Rotate((-90, 90), (-90, 90), (0, 0), p=1.0),],
            #     additional_targets={'image0': 'image', 'image1': 'image'},
            #     p=1.0)
            # data = {'image': img, 'mask': heatmap[0,:,:,:]}
            # aug_data = aug(**data)
            # img, heatmap = aug_data['image'], aug_data['mask']
            
            # random_rotation = randint(-15,-15)
            # img = F.rotate(angle = random_rotation,img=img,fill=-1)
            # heatmap = F.rotate(angle = random_rotation,img=heatmap,fill=0)
            
        #Convert to tensor
        img = torch.from_numpy(img)
        heatmap = torch.from_numpy(heatmap)

        #Add one dimension to image to set #channels = 1. The heatmap is already in 4 dimensions with 8 channels from heatmap_generation script
        img = img.unsqueeze(0) 
            
        return img, heatmap, subject
    




class LoadValidationData(Dataset):

    def __init__(self, img_dir, heatmap_dir, transform=None):
        self.img_dir = img_dir
        self.heatmap_dir = heatmap_dir
        self.images = os.listdir(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir,self.images[index])
        heatmap_path = os.path.join(self.heatmap_dir,self.images[index].replace('img.nii.gz','heatmap.nii.gz'))

        #Load image and heatmap
        img = nib.load(img_path)
        heatmap = nib.load(heatmap_path)
        #Get data shape
        dim1, dim2, dim3 =  img.header.get_data_shape()
        #Convert to right format
        img = np.asanyarray(img.dataobj, dtype=np.float32)
        heatmap = np.asanyarray(heatmap.dataobj, dtype=np.float32)

        #Crop images randomly if z-axis is too long
        if dim3 > 128:
            no_pathces = np.ceil(dim3/128).astype(int)
            list_of_images = []
            list_of_heatmaps = []
            start_voxel = 0
            for i in range(no_pathces):
                end_voxel = start_voxel + 128
                cropped_img = img[:,:,start_voxel:end_voxel]
                cropped_heatmap = heatmap[:,:,start_voxel:end_voxel]
                #Convert to tensor
                cropped_img = torch.from_numpy(cropped_img)
                cropped_heatmap = torch.from_numpy(cropped_heatmap)
                #Add one dimension to image to set #channels = 1. The heatmap is already in 4 dimensions with 8 channels from heatmap_generation script
                cropped_img = cropped_img.unsqueeze(0) 
                #Save to list
                list_of_images.append(cropped_img)
                list_of_heatmaps.append(cropped_heatmap)
                #Update start-boxel
                start_voxel = end_voxel - 96

        #SUBJECT
        subject = self.images[index].split("_")[0]

        #Fjern dette for validation
        if self.transform is not None:
            random_rotation = randint(-15,-15)
            img = F.rotate(angle = random_rotation,img=img,fill=-1)
            heatmap = F.rotate(angle = random_rotation,img=heatmap,fill=0)
            
        return img, heatmap, subject