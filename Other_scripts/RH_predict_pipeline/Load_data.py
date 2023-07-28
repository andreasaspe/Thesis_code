import os
import numpy as np
from torch.utils.data import Dataset
import nibabel as nib
import torch

class LoadData_spine(Dataset):

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
        subject = self.images[index].split(".")[0]
        return img, subject