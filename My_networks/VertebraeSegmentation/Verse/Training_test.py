#General imports
import os
import sys
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.transforms as transforms
from torch import linalg as LA
import wandb
#My own documents
from my_plotting_functions import *
from Create_dataset import LoadData
from my_data_utils import Predict, gaussian_kernel_3d
from VertebraeSegmentationNet import *
#from VertebraeSegmentationNet_batchnormdropout import *

#Define paramters
parameters_dict = {
    'epochs': 3000,
    'learning_rate': 1e-5, #1e-5, # 1e-8
    'weight_decay': 5e-4,
    'batch_size': 1,
    'dropout': 0.2
}


#For everything
#gpu-cluster
#Training
img_dir_training = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeSegmentation/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_training_prep/img' #'/scratch/s174197/data/Verse20/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20_training_prep/img' #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data\img'
heatmap_dir_training = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeSegmentation/Verse20_training_prep/heatmaps'
msk_dir_training = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeSegmentation/Verse20_training_prep/msk'



#Unpack parameters
num_epochs = parameters_dict['epochs']
lr = parameters_dict['learning_rate']
wd = parameters_dict['weight_decay']
batch_size = parameters_dict['batch_size']
dropout = parameters_dict['dropout']


#Load data
VerSe_train = LoadData(img_dir=img_dir_training, heatmap_dir=heatmap_dir_training, msk_dir = msk_dir_training , transform='elastic') #elastic, rotation, both or None
train_loader = DataLoader(VerSe_train, batch_size=batch_size,
                        shuffle=False, num_workers=0) #SET TO True!


#Define model
model = Unet3D(dropout).double()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #device = torch.device('cpu')
model.to(device)

#Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay=wd) #Oprindeligt stod der 0.0001
#optimizer = optim.SGD(params=model.parameters(), lr = lr, weight_decay=wd, momentum=0.1, nesterov=True)


train_loss = []
val_loss = []

step = -1

#Train loop
for epoch in range(num_epochs):

    print("Epoch nr.: "+str(epoch))
    train_loss_batches = []
    
    #Train model
    model.train()

    #Batching through data
    for inputs, targets, subject in tqdm(train_loader): #tqdm(train_loader)
        subject = subject[0]
        inputs = inputs.squeeze().detach().cpu().numpy()
        img_data = inputs[0]
        heatmap_data = inputs[1]
        msk_data = targets.squeeze().detach().cpu().numpy()
        
        # show_slices_dim1(img_data, subject)
        # show_mask_dim1(msk_data, subject)
        show_mask_img_dim1(img_data, msk_data, subject)
        # show_heatmap_img_dim1(img_data, heatmap_data, subject)