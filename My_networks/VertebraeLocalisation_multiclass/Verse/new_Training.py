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
from Create_dataset import *
from my_data_utils import Predict, gaussian_kernel_3d
#from VertebraeLocalisationNet import *
from new_VertebraeLocalisationNet import *


#Define paramters
parameters_dict = {
    'epochs': 3000,
    'learning_rate': 0.00001, #1e-5, # 1e-8
    'weight_decay': 0.00001,
    'batch_size': 1,
    'dropout': 0.3
}


#For everything
#gpu-cluster
img_dir_training = '/scratch/s174197/data/Verse20/VertebraeLocalisation/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_training_prep/img' #'/scratch/s174197/data/Verse20/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20_training_prep/img' #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data\img'
heatmap_dir_training = '/scratch/s174197/data/Verse20/VertebraeLocalisation/Verse20_training_heatmaps'
img_dir_validation = '/scratch/s174197/data/Verse20/VertebraeLocalisation/Verse20_validation_prep/img'
heatmap_dir_validation = '/scratch/s174197/data/Verse20/VertebraeLocalisation/Verse20_validation_heatmaps'
checkpoint_dir = '/scratch/s174197/data/Checkpoints/VertebraeLocalisation/NEW' #'/Users/andreasaspe/Library/Mobile Documents/com~apple~CloudDocs/DTU/12.semester/Thesis/My_code/My_networks/Spine_Localisation/Checkpoints'
checkpoint_filename = 'Epoch' #No underscore after this

#mac
# img_dir_training = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_training_prep/img' #'/scratch/s174197/data/Verse20/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20_training_prep/img' #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data\img'
# heatmap_dir_training = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation/Verse20_training_heatmaps'
# img_dir_validation = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation/Verse20_validation_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_training_prep/img' #'/scratch/s174197/data/Verse20/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20_training_prep/img' #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data\img'
# heatmap_dir_validation = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation/Verse20_validation_heatmaps'




#Load data
VerSe_train = LoadData(img_dir=img_dir_training, heatmap_dir=heatmap_dir_training,transform=None)
train_loader = DataLoader(VerSe_train, batch_size=1,
                        shuffle=True, num_workers=0) #SET TO True!
VerSe_val = LoadFullData(img_dir=img_dir_validation, heatmap_dir=heatmap_dir_validation)
val_loader = DataLoader(VerSe_val, batch_size=1,
                        shuffle=False, num_workers=0) #SET TO False!


#Unpack parameters
num_epochs = parameters_dict['epochs']
lr = parameters_dict['learning_rate']
wd = parameters_dict['weight_decay']
batch_size = parameters_dict['batch_size']
dropout = parameters_dict['dropout']

#Start wand.db
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="VertebraeLocalisation",
#     entity='andreasaspe',
    
#     # track hyperparameters and run metadata
#     config={
#     "learning_rate": lr,
#     "epochs": num_epochs,
#     'weight_decay': wd,
#     'batch_size': batch_size,
#     'drop_out': dropout,
#     'Checkpoint_filename': checkpoint_filename,
#     }
# )



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
    for inputs, targets, subject in tqdm(train_loader): #tqdm(train_loader)    
        #if subject[0] == 'sub-verse537': #Godt subject!
        #show_slices_dim1(inputs[0,0,:,:,:],subject[0],no_slices=40)
        #show_heatmap_dim1(targets[0,0,:,:,:], subject[0])
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


    # #Back to epoch loop
    avg_loss_train = np.mean(train_loss_batches)
    train_loss.append(avg_loss_train)
    print("          EPOCH TRAINING LOSS: "+str(avg_loss_train))


    # # Compute accuracies on validation set.
    model.eval() #Set to evaluation
    val_loss_batches = []
    with torch.no_grad():
        for i, (img, heatmap, inputs_list, targets_list, start_end_voxels, subject) in enumerate(tqdm(val_loader)):
            _, dim1, dim2, dim3 = img.shape
            outputs_list = []
            for i in range(len(inputs_list)):
                #Initialise vector
                full_output_temp = torch.ones((1,8,dim1,dim2,dim3))*(-1000)
                #Unpack targets and inputs and get predictions
                inputs, targets = inputs_list[i].to(device), targets_list[i].to(device)
                output = model(inputs)
                #Get start and end voxel
                start_voxel = start_end_voxels[i][0].item()
                end_voxel = start_end_voxels[i][1].item()
                #Put into the output_tensor
                full_output_temp[:,:,:,:,start_voxel:end_voxel+1] = output
                #Append to list
                outputs_list.append(full_output_temp)
            #Taking the maximum response
            final_output = torch.max(*outputs_list)

            loss = loss_fn(final_output, heatmap)
            if loss.item() > 0:
                allisgood=1
            else:
                print("Okay, nu sker der noget!!")
            # Save loss
            val_loss_batches.append(loss.item())

    avg_loss_validation = np.mean(val_loss_batches)
    val_loss.append(avg_loss_validation)
    print("          EPOCH VALIDATION LOSS: "+str(avg_loss_validation))

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
    }

    #torch.save(checkpoint, os.path.join(checkpoint_dir,str(checkpoint_filename)+'_batchsize'+str(batch_size)+'_lr'+str(lr)+'_wd'+str(wd)+'.pth'))
    torch.save(checkpoint, os.path.join(checkpoint_dir,str(checkpoint_filename)+str(epoch)+'.pth'))

    #Log in wandb
    #wandb.log({"Train_loss": avg_loss_train, "Validation_loss": avg_loss_validation})






















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