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



#For everything
#gpu-cluster
img_dir_training = '/scratch/s174197/data/Verse20/VertebraeLocalisation/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_training_prep/img' #'/scratch/s174197/data/Verse20/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20_training_prep/img' #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data\img'
heatmap_dir_training = '/scratch/s174197/data/Verse20/VertebraeLocalisation/Verse20_training_heatmaps'
img_dir_validation = '/scratch/s174197/data/Verse20/VertebraeLocalisation/Verse20_validation_prep/img'
heatmap_dir_validation = '/scratch/s174197/data/Verse20/VertebraeLocalisation/Verse20_validation_heatmaps'

#mac
# img_dir_training = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_training_prep/img' #'/scratch/s174197/data/Verse20/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20_training_prep/img' #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data\img'
# heatmap_dir_training = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation/Verse20_training_heatmaps'
# img_dir_validation = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation/Verse20_validation_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_training_prep/img' #'/scratch/s174197/data/Verse20/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20_training_prep/img' #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data\img'
# heatmap_dir_validation = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation/Verse20_validation_heatmaps'


#Sweeping
# method
sweep_config = {
    'method': 'random'
}

#Define paramters
parameters_dict = {
    'learning_rate': {
        'distribution': 'log_uniform_values',
        'min': 1e-8,
        'max': 1e-3
    },
    'weight_decay': {
        'distribution': 'log_uniform_values',
        'min': 0.00001,
        'max': 0.1
    },
    'dropout': {
        'values': [0.05, 0.1, 0.2, 0.3, 0.4]
    },
    'momentum': {
        'distribution': 'log_uniform_values',
        'min': 0.1,
        'max': 0.9
    },
    'optimizer': {
        'values': ['adam','sgd']
    },
    'nesterov_parameter':{
        'values': [True,False]
    }
}

sweep_config['parameters'] = parameters_dict
sweep_id = wandb.sweep(sweep_config, project="VertebraeLocalisation", entity='andreasaspe')




transform = transforms.Compose([
    transforms.RandomRotation(10)
])



#Load data
VerSe_train = LoadData(img_dir=img_dir_training, heatmap_dir=heatmap_dir_training,transform=transform)
train_loader = DataLoader(VerSe_train, batch_size=1,
                        shuffle=True, num_workers=0) #SET TO True!
VerSe_val = LoadData(img_dir=img_dir_validation, heatmap_dir=heatmap_dir_validation)
val_loader = DataLoader(VerSe_val, batch_size=1,
                        shuffle=False, num_workers=0) #SET TO False!




def main(config=None):
    with wandb.init(config=config):
        config = wandb.config

        #Unpack parameters
        lr = config.learning_rate
        wd = config.weight_decay
        dropout = config.dropout
        momentum = config.momentum
        optimizer = config.optimizer
        nesterov_parameter = config.nesterov_parameter

        num_epochs = 50

        #Define model
        model = VertebraeLocalisationNet(dropout)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #device = torch.device('cpu')
        model.to(device)


        #Define loss function and optimizer
        loss_fn = nn.MSELoss()
        if optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay=wd) #Oprindeligt stod der 0.0001
        else:
            optimizer = optim.SGD(params=model.parameters(), lr = lr, weight_decay=wd, momentum=momentum, nesterov=nesterov_parameter)


        train_loss = []
        val_loss = []


        #Train loop
        for epoch in range(num_epochs):

            print("Epoch nr.: "+str(epoch))
            train_loss_batches = []
            
            #Train model
            model.train()

            #Batching through data
            for inputs, targets, subject in tqdm(train_loader): #tqdm(train_loader)
            
                #if subject[0] == 'sub-verse537': #Godt subject!
                #show_slices_dim1(inputs[0,0,:,:,:],subject[0],no_slices=3)
                #show_heatmap_img_dim1(inputs[0,0,:,:,:], targets[0,0,:,:,:], subject[0])


                #Send to device
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass, compute gradients, perform one training step.
                output = model(inputs) #Output
                loss = loss_fn(output, targets) #Loss
                optimizer.zero_grad() #Clean up gradients
                loss.backward() #Compute gradients
                optimizer.step () #Clean up gradients

                # Save loss
                train_loss_batches.append(loss.item())


            #Back to epoch loop
            avg_loss_train = np.mean(train_loss_batches)
            train_loss.append(avg_loss_train)
            print("          EPOCH TRAINING LOSS: "+str(avg_loss_train))


            # Compute accuracies on validation set.
            model.eval() #Set to evaluation
            val_loss_batches = []
            with torch.no_grad():
                for inputs, centroids, _ in tqdm(val_loader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    output = model(inputs)
                    loss = loss_fn(output, targets)
                    # Save loss
                    val_loss_batches.append(loss.item())

        
            avg_loss_validation = np.mean(val_loss_batches)
            val_loss.append(avg_loss_validation)
            print("          EPOCH VALIDATION LOSS: "+str(avg_loss_validation))


            #Log in wandb
            wandb.log({"Train_loss": avg_loss_train, "Validation_loss": avg_loss_validation})


wandb.agent(sweep_id, main, count=200)


if __name__ == "__main__":
    main()
















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