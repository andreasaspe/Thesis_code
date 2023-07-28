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
#My moduiles
from data_utilities import load_centroids
from my_data_utils import gaussian_kernel_3d


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
            heatmap = heatmap[:,:,bottom_voxel:top_voxel+1]

        #Convert to tensor
        img = torch.from_numpy(img)
        heatmap = torch.from_numpy(heatmap)

        #Add one channel dimension
        img = img.unsqueeze(0) 
        heatmap = heatmap.unsqueeze(0) 

        #SUBJECT
        subject = self.images[index].split("_")[0]

        if self.transform is not None:
            random_rotation = randint(-15,-15)
            img = F.rotate(angle = random_rotation,img=img,fill=-1)
            heatmap = F.rotate(angle = random_rotation,img=heatmap,fill=0)
            
        return img, heatmap, subject
    



class LoadFullData(Dataset):

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
            no_patches = np.ceil(dim3/128).astype(int)
            list_of_images = []
            list_of_heatmaps = []
            start_end_voxels = [] #It is actual voxel start and end values
            start_voxel = 0
            for i in range(no_patches):

                #Find start and end voxel
                end_voxel = start_voxel + 127
                if end_voxel + 1 > dim3:
                    start_voxel = dim3-128
                    end_voxel = dim3-1
                    
                start_end_voxels.append((start_voxel,end_voxel))
                    
                # if i != 0:
                #     overlap_voxels.append((start_voxel,former_endvoxel)) #Start voxel here is new start voxel
                    
                # former_endvoxel = end_voxel
                
                #Image
                cropped_img = img[:,:,start_voxel:end_voxel+1]
                #Convert to tensor
                cropped_img = torch.from_numpy(cropped_img)
                #Add one dimension to image to set #channels = 1
                cropped_img = cropped_img.unsqueeze(0) 
                #Save to list
                list_of_images.append(cropped_img)

                #Heatmap
                heatmap_temp = np.zeros((96,96,128))
                for j in range(8):
                    cropped_heatmap = heatmap[:,:,start_voxel:end_voxel+1]
                    heatmap_temp[:,:,:] = cropped_heatmap
                #To tensor
                heatmap_temp = torch.from_numpy(heatmap_temp)
                #Add one dimension to heatmap to set #channels = 1
                cropped_img = cropped_img.unsqueeze(0) 
                #Save to list
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
            
        return img, heatmap, list_of_images, list_of_heatmaps, start_end_voxels, subject
    
    



    