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
import random
import scipy
import SimpleITK as sitk
import elasticdeform
import random
#My moduiles
from data_utilities import load_centroids
from my_data_utils import gaussian_kernel_3d


#For multiple batch-sizes! Source is from here: https://stackoverflow.com/questions/43922198/how-to-rotate-a-3d-image-by-a-random-angle-in-python
# def random_rotation_3d(batch, max_angle):
#     """ Randomly rotate an image by a random angle (-max_angle, max_angle).

#     Arguments:
#     max_angle: `float`. The maximum rotation angle.

#     Returns:
#     batch of rotated 3D images
#     """
#     size = batch.shape
#     batch = np.squeeze(batch)
#     batch_rot = np.zeros(batch.shape)
#     for i in range(batch.shape[0]):
#         if bool(random.getrandbits(1)):
#             image1 = np.squeeze(batch[i])
#             # rotate along z-axis
#             angle = random.uniform(-max_angle, max_angle)
#             image2 = scipy.ndimage.interpolation.rotate(image1, angle, mode='nearest', axes=(0, 1), reshape=False)

#             # rotate along y-axis
#             angle = random.uniform(-max_angle, max_angle)
#             image3 = scipy.ndimage.interpolation.rotate(image2, angle, mode='nearest', axes=(0, 2), reshape=False)

#             # rotate along x-axis
#             angle = random.uniform(-max_angle, max_angle)
#             batch_rot[i] = scipy.ndimage.interpolation.rotate(image3, angle, mode='nearest', axes=(1, 2), reshape=False)
#             #                print(i)
#         else:
#             batch_rot[i] = batch[i]
#     return batch_rot.reshape(size)



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
        _, _, dim3 =  img.header.get_data_shape()
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

        #Augmentation
        if self.transform is not None:
            self.elastic = 0
            self.rotation = 0
            #Test cases
            if self.transform == 'elastic':
                #Probability for elastic deformation
                p_elastic = 0.5
                self.elastic = np.random.choice(np.arange(2), p=[ 1-p_elastic , p_elastic])    
            
            if self.transform == 'rotation':
                #Probability of doing rotiation    
                p_rotation = 0.5
                self.rotation = np.random.choice(np.arange(2), p=[ 1-p_rotation , p_rotation])
            
            if self.transform == 'both':
                #Probability for elastic deformation
                p_elastic = 0.5
                self.elastic = np.random.choice(np.arange(2), p=[ 1-p_elastic , p_elastic])
                #Probability of doing rotiation    
                p_rotation = 0.5 
                self.rotation = np.random.choice(np.arange(2), p=[ 1-p_rotation , p_rotation])

            #Perform
            if self.elastic:
                # print("Elastic deformation is performed")
                img = elasticdeform.deform_random_grid(img, sigma=2, points=6, cval=-1, order=3)
            if self.rotation:
                # print("Rotation is performed")
                #Probability for rotation
                angle1 = random.uniform(-15,15)
                angle2 = random.uniform(-15,15)
                angle3 = random.uniform(-15,15)
                img = scipy.ndimage.rotate(img, angle1, order=3, axes=(1,2), cval=-1, reshape=False)
                heatmap = scipy.ndimage.rotate(heatmap, angle1, order=3, axes=(1,2), cval=0, reshape=False)
                
                img = scipy.ndimage.rotate(img, angle2, order=3, axes=(0,2), cval=-1, reshape=False)
                heatmap = scipy.ndimage.rotate(heatmap, angle2, order=3, axes=(0,2), cval=0, reshape=False)

                img = scipy.ndimage.rotate(img, angle3, order=3, axes=(0,1), cval=-1, reshape=False)
                heatmap = scipy.ndimage.rotate(heatmap, angle3, order=3, axes=(0,1), cval=0, reshape=False)
            
            
        #Convert to tensor
        img = torch.from_numpy(img)
        heatmap = torch.from_numpy(heatmap)

        #Add one channel dimension
        img = img.unsqueeze(0) 
        heatmap = heatmap.unsqueeze(0) 

        #SUBJECT
        subject = self.images[index].split("_")[0]


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
            list_of_images = []
            list_of_heatmaps = []
            start_end_voxels = [] #It is actual voxel start and end values
            #Start values
            start_voxel = 0
            finished = False
            while not finished:
                #Find end voxel and probably renew start voxel if finished
                end_voxel = start_voxel + 127
                if end_voxel + 1 > dim3:
                    start_voxel = dim3-128
                    end_voxel = dim3-1
                    
                start_end_voxels.append((start_voxel,end_voxel))

                #Check if we should stop after this iteration
                if end_voxel == dim3 - 1: #Will be the case if finished
                    finished = True
                    
                #Image
                cropped_img = img[:,:,start_voxel:end_voxel+1]
                #Convert to tensor
                cropped_img = torch.from_numpy(cropped_img)
                #Add one dimension to image to set #channels = 1
                cropped_img = cropped_img.unsqueeze(0) 
                #Save to list
                list_of_images.append(cropped_img)

                #Heatmap
                cropped_heatmap = heatmap[:,:,start_voxel:end_voxel+1]
                #Convert to tensor
                cropped_heatmap = torch.from_numpy(cropped_heatmap)
                #Add one dimension to heatmap to set #channels = 1
                cropped_heatmap = cropped_heatmap.unsqueeze(0) 
                #Save to list
                list_of_heatmaps.append(cropped_heatmap)

                #Update start-boxel (with overlap of 96 voxels.)
                start_voxel = end_voxel - 95
        else: #The dimensions are exactly right and there is only one image in list. No need for cropping.
            #Image
            cropped_img = img
            cropped_img = torch.from_numpy(cropped_img)
            cropped_img = cropped_img.unsqueeze(0)
            list_of_images = [cropped_img]
            #Heatmap
            cropped_heatmap = heatmap
            cropped_heatmap = torch.from_numpy(cropped_heatmap)
            cropped_heatmap = cropped_heatmap.unsqueeze(0)
            list_of_heatmaps = [cropped_heatmap]
            #Start end voxels
            start_voxel = 0
            end_voxel = 127
            start_end_voxels = [(start_voxel,end_voxel)]

        #SUBJECT
        subject = self.images[index].split("_")[0]

        #Fjern dette for validation
        if self.transform is not None:
            random_rotation = randint(-15,-15)
            img = F.rotate(angle = random_rotation,img=img,fill=-1)
            heatmap = F.rotate(angle = random_rotation,img=heatmap,fill=0)
            
        return img, heatmap, list_of_images, list_of_heatmaps, start_end_voxels, subject
    
    



    












            # if self.transform is not None:
        #     #Probability for elastic deformation
        #     # elastic = np.random.choice(np.arange(2), p=[0.5,0.5])
        #     rotation = np.random.choice(np.arange(2), p=[0.5,0.5])
        #     # if elastic:
        #     #     print("Elastic deformation is performed")
        #     #     img = elasticdeform.deform_random_grid(img, sigma=1.5, points=10,cval = -1)
        #     if rotation:
        #         # print("Rotation is performed")
        #         #Probability for rotation
        #         angle1 = random.uniform(-15,15)
        #         angle2 = random.uniform(-15,15)
        #         angle3 = random.uniform(-15,15)
        #         img = scipy.ndimage.interpolation.rotate(img, angle1, mode='nearest', axes=(1,2), cval=-1, reshape=False)
        #         heatmap = scipy.ndimage.interpolation.rotate(heatmap, angle1, mode='nearest', axes=(1,2), cval=-1, reshape=False)
                
        #         img = scipy.ndimage.interpolation.rotate(img, angle2, mode='nearest', axes=(0,2), cval=-1, reshape=False)
        #         heatmap = scipy.ndimage.interpolation.rotate(heatmap, angle2, mode='nearest', axes=(0,2), cval=-1, reshape=False)

        #         img = scipy.ndimage.interpolation.rotate(img, angle3, mode='nearest', axes=(0,1), cval=-1, reshape=False)
        #         heatmap = scipy.ndimage.interpolation.rotate(heatmap, angle3, mode='nearest', axes=(0,1), cval=-1, reshape=False)

        #     # random_rotation = randint(-15,-15)
        #     # img = F.rotate(angle = random_rotation,img=img,fill=-1)
        #     # heatmap = F.rotate(angle = random_rotation,img=heatmap,fill=0)