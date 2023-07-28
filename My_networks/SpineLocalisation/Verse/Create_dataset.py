#Windows shortcuts
#Comment: Ctrl+k+c (c for comment)
#Uncomment: Ctrl+k+u (u for uncomment)

import os
import numpy as np
from torch.utils.data import Dataset
import nibabel as nib
import torch
from random import *
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import scipy
import elasticdeform
import random



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
        #img
        img = nib.load(img_path)
        img = np.asanyarray(img.dataobj, dtype=np.float32)
        #img = img.get_fdata()

        #img = img.double()
        heatmap = nib.load(heatmap_path)
        heatmap = np.asanyarray(heatmap.dataobj, dtype=np.float32)
        #heatmap = heatmap.get_fdata()
        

        # if self.transform is not None:
        #     random_rotation = randint(-15,15)
        #     img = F.rotate(angle = random_rotation,img=img)
        #     heatmap = F.rotate(angle = random_rotation,img=heatmap)
            
        #Augmentation
        if self.transform is not None:
            #Test cases
            if self.transform == 'elastic':
                #Probability for elastic deformation
                p_elastic = 0.5
                elastic = np.random.choice(np.arange(2), p=[ 1-p_elastic , p_elastic])
            else:
                elastic = 0
            
            if self.transform == 'rotation':
                #Probability of doing rotiation    
                p_rotation = 0.5 
                rotation = np.random.choice(np.arange(2), p=[ 1-p_rotation , p_rotation])
            else:
                rotation = 0
            
            if self.transform == 'both':
                #Probability for elastic deformation
                p_elastic = 0.5
                elastic = np.random.choice(np.arange(2), p=[ 1-p_elastic , p_elastic])
                #Probability of doing rotiation    
                p_rotation = 0.5 
                rotation = np.random.choice(np.arange(2), p=[ 1-p_rotation , p_rotation])
            else:
                elastic = 0
                rotation = 0

            #Perform
            if elastic:
                # print("Elastic deformation is performed")
                img = elasticdeform.deform_random_grid(img, sigma=2, points=6, cval=-1, order=3)
            if rotation:
                # print("Rotation is performed")
                angle1 = random.uniform(-15,15)
                angle2 = random.uniform(-15,15)
                angle3 = random.uniform(-15,15)
                img = scipy.ndimage.rotate(img, angle1, order=3, axes=(1,2), cval=-1, reshape=False)
                heatmap = scipy.ndimage.rotate(heatmap, angle1, order=3, axes=(1,2), cval=0, reshape=False)
                
                img = scipy.ndimage.rotate(img, angle2, order=3, axes=(0,2), cval=-1, reshape=False)
                heatmap = scipy.ndimage.rotate(heatmap, angle2, order=3, axes=(0,2), cval=0, reshape=False)

                img = scipy.ndimage.rotate(img, angle3, order=3, axes=(0,1), cval=-1, reshape=False)
                heatmap = scipy.ndimage.rotate(heatmap, angle3, order=3, axes=(0,1), cval=0, reshape=False)
            
        
        img = torch.from_numpy(img)
        img = img.unsqueeze(0) 
        heatmap = torch.from_numpy(heatmap)
        heatmap = heatmap.unsqueeze(0) 

        #heatmap = heatmap.double()
        subject = self.images[index].split("_")[0]
            
        return img, heatmap, subject

class LoadData_notarget(Dataset):

    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.images = os.listdir(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir,self.images[index])
        img = nib.load(img_path)
        img = np.asanyarray(img.dataobj, dtype=np.float32)
        img = torch.from_numpy(img)
        img = img.unsqueeze(0) 
        subject = self.images[index].split("_")[0]
        return img, subject


# class FaceLandmarksDataset(Dataset):
#     """Face Landmarks dataset."""

#     def __init__(self, csv_file, root_dir, transform=None):
#         """
#         Arguments:
#             csv_file (string): Path to the csv file with annotations.
#             root_dir (string): Directory with all the images.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         self.landmarks_frame = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.landmarks_frame)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         img_name = os.path.join(self.root_dir,
#                                 self.landmarks_frame.iloc[idx, 0])
#         image = io.imread(img_name)
#         landmarks = self.landmarks_frame.iloc[idx, 1:]
#         landmarks = np.array([landmarks])
#         landmarks = landmarks.astype('float').reshape(-1, 2)
#         sample = {'image': image, 'landmarks': landmarks}

#         if self.transform:
#             sample = self.transform(sample)

#         return sample