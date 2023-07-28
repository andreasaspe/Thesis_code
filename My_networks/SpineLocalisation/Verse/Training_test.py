#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 10:42:30 2023

@author: andreasaspe
"""

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
#My own documents
from my_plotting_functions import *
from SpineLocalisationNet import *
from Create_dataset import LoadData
from my_data_utils import Predict


#DEFINE STUFF
parameters_dict = {
    'epochs': 
        1,
    'learning_rate': 
        1e-4,
    'weight_decay': 
         5e-4
}

transform = transforms.Compose([
    transforms.RandomRotation(50),
])

#For everything
img_dir_training = '/scratch/s174197/data/Verse20/SpineLocalisation/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_training_prep/img' #'/scratch/s174197/data/Verse20/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20_training_prep/img' #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data\img'
heatmap_dir_training = '/scratch/s174197/data/Verse20/SpineLocalisation/Verse20_training_heatmaps' #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_training_heatmaps' #'/scratch/s174197/data/Verse20/Verse20_training_heatmaps' #'/Users/andreasaspe/Documents/Data/Verse20_training_heatmaps' #r'C:\Users\PC\Documents\Andreas_s174197\heatmaps'


#Define model
model = Unet3D()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #device = torch.device('cpu')
model.to(device)

#Dummy pass
# out = model(torch.rand(1,1,128,64,64,device=device))
# print("Output shape:", out.size())

#Unpack parameters
total_epochs = parameters_dict['epochs']
lr = parameters_dict['learning_rate']
wd = parameters_dict['weight_decay']

#Define values
num_epochs = total_epochs
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay=wd) #Oprindeligt stod der 0.0001
batch_size = 1

#Load data
VerSe_train = LoadData(img_dir=img_dir_training,heatmap_dir=heatmap_dir_training,transform=transform)
train_loader = DataLoader(VerSe_train, batch_size=batch_size,
                        shuffle=True, num_workers=0)



#Train model
model.train()


train_loss = []
val_loss = []

#Train loop
for epoch in range(num_epochs):
    
    print("Epoch nr.: "+str(epoch))
    train_loss_batches = []
    
    #Batching through data
    for inputs, targets, _ in tqdm(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        
        # inputs = inputs.squeeze(0,1).numpy()
        # show_slices_dim1(inputs,no_slices=40)
