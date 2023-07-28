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
#from VertebraeLocalisationNet import *
from new_VertebraeLocalisationNet import *
#from VertebraeLocalisationNet_newdropout import *

#Define paramters
parameters_dict = {
    'epochs': 3000,
    'learning_rate': 1e-5, #1e-5, # 1e-8
    'weight_decay': 0.0001,
    'batch_size': 1,
    'dropout': 0.0
}



#For everything
#gpu-cluster
# img_dir_training = '/scratch/s174197/data/Verse20/VertebraeLocalisation2/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_training_prep/img' #'/scratch/s174197/data/Verse20/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20_training_prep/img' #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data\img'
# heatmap_dir_training = '/scratch/s174197/data/Verse20/VertebraeLocalisation2/Verse20_training_heatmaps'
# img_dir_validation = '/scratch/s174197/data/Verse20/VertebraeLocalisation2/Verse20_validation_prep/img'
# heatmap_dir_validation = '/scratch/s174197/data/Verse20/VertebraeLocalisation2/Verse20_validation_heatmaps'
# checkpoint_dir = '/scratch/s174197/data/Checkpoints/VertebraeLocalisation2/First_try' #'/Users/andreasaspe/Library/Mobile Documents/com~apple~CloudDocs/DTU/12.semester/Thesis/My_code/My_networks/Spine_Localisation/Checkpoints'
# checkpoint_filename = 'No_dropout_newinitialisation' #No underscore after this

#mac
img_dir_training = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation2/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_training_prep/img' #'/scratch/s174197/data/Verse20/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20_training_prep/img' #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data\img'
heatmap_dir_training = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation2/Verse20_training_heatmaps'
img_dir_validation = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation2/Verse20_validation_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_training_prep/img' #'/scratch/s174197/data/Verse20/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20_training_prep/img' #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data\img'
heatmap_dir_validation = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation2/Verse20_validation_heatmaps'


#Unpack parameters
num_epochs = parameters_dict['epochs']
lr = parameters_dict['learning_rate']
wd = parameters_dict['weight_decay']
batch_size = parameters_dict['batch_size']
dropout = parameters_dict['dropout']


#Load data
VerSe_train = LoadData(img_dir=img_dir_training, heatmap_dir=heatmap_dir_training,transform='elastic')
train_loader = DataLoader(VerSe_train, batch_size=batch_size,
                        shuffle=False, num_workers=0) #SET TO True!
VerSe_val = LoadData(img_dir=img_dir_validation, heatmap_dir=heatmap_dir_validation)
val_loader = DataLoader(VerSe_val, batch_size=batch_size,
                        shuffle=False, num_workers=0) #SET TO False!



#Define model
model = VertebraeLocalisationNet(dropout)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #device = torch.device('cpu')
model.to(device)
print(device)


#Define loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay=wd) #Oprindeligt stod der 0.0001
#optimizer = optim.SGD(params=model.parameters(), lr = lr, weight_decay=wd, momentum=0.1, nesterov=True)


train_loss = []
val_loss = []


#Train loop
for epoch in range(num_epochs):

    print("Epoch nr.: "+str(epoch))
    train_loss_batches = []
    
    #Train model
    model.train()

    #Batching through data
    for i, (inputs, targets, subject) in tqdm(enumerate(train_loader)): #tqdm(train_loader)
    
        show_slices_dim1(inputs[0,0,:,:,:],subject[0],no_slices=40)
        # print(i)
        continue

        #if subject[0] == 'sub-verse537': #Godt subject!
        # show_slices_dim1(inputs[0,0,:,:,:],subject[0],no_slices=40)
        # show_heatmap_dim1(targets[0,0,:,:,:], subject[0])
        show_heatmap_img_dim1(inputs[0,0,:,:,:], targets[0,0,:,:,:], subject[0])
        
        # show_slices_dim2(inputs[0,0,:,:,:],subject[0],no_slices=40)
        # show_slices_dim3(inputs[0,0,:,:,:],subject[0],no_slices=40)
        #show_heatmap_dim1(targets[0,0,:,:,:], subject[0])
        # show_heatmap_img_dim1(inputs[0,0,:,:,:], targets[0,0,:,:,:], subject[0])
        

        #Send to device
        inputs, targets = inputs.to(device), targets.to(device)




    # # Compute accuracies on validation set.
    model.eval() #Set to evaluation
    with torch.no_grad():
        for inputs, targets, _ in tqdm(val_loader):

            inputs, targets = inputs.to(device), targets.to(device)
















        # print("Epoch is: "+str(epoch))
        # print(loss_fn.sigma)




# for name, param in vae.named_parameters():
    # if param.requires_grad:
    #     if name == "log_sigma":
    #       print(param.exp())



    # if subject[0] == 'sub-verse504': #sub-verse646
    #     for i in range(8):
    #         show_heatmap_dim1(targets[0,i,:,:,:], subject[0], no_slices=40)
 
    #show_heatmap_dim1(targets[0,0,:,:,:], no_slices=40)


# class CustomLoss(nn.Module):
    
#     def __init__(self):
#         super(CustomLoss, self).__init__()

#     def forward(self, output, target):
#         target = torch.LongTensor(target)
#         criterion = nn.CrossEntropyLoss()
#         loss = criterion(output, target)
#         mask = target == 9
#         high_cost = (loss * mask.float()).mean()
#         return loss + high_cost