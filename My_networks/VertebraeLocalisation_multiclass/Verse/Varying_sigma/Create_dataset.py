#Windows shortcuts
#Comment: Ctrl+k+c (c for comment)
#Uncomment: Ctrl+k+u (u for uncomment)

import os
import numpy as np
from torch.utils.data import Dataset
import nibabel as nib
import torch
#My moduiles
from data_utilities import load_centroids
from my_data_utils import gaussian_kernel_3d

class LoadData(Dataset):

    def __init__(self, img_dir, ctd_dir, transform=None):
        self.img_dir = img_dir
        self.ctd_dir = ctd_dir
        self.transform = transform
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir,self.images[index])
        ctd_path = os.path.join(self.ctd_dir,self.images[index].replace('img.nii.gz','ctd.json'))

        #LOAD CENTROIDS
        ctd_list = load_centroids(ctd_path)


        #IMAGE
        #Load image
        img = nib.load(img_path)
        #Get data shape
        dim1, dim2, dim3 =  img.header.get_data_shape()
        #Convert to right format
        img = np.asanyarray(img.dataobj, dtype=np.float32)
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
            #Translate centroids
            for ctd in ctd_list[1:]:
                ctd[3] = ctd[3]-bottom_voxel

        img = torch.from_numpy(img)
        img = img.unsqueeze(0) 

        #SUBJECT
        subject = self.images[index].split("_")[0]

        if self.transform:
            img = self.transform(img)
            
        return img, ctd_list, subject

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