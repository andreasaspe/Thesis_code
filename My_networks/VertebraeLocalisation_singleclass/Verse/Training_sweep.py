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


#For everything
#gpu-cluster
img_dir_training = '/scratch/s174197/data/Verse20/VertebraeLocalisation2/Verse20_training_prep/img' #'/scratch/s174197/data/Verse20/VertebraeLocalisation2/Verse20_training_prep_alldata/img' #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_training_prep/img' #'/scratch/s174197/data/Verse20/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20_training_prep/img' #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data\img'
heatmap_dir_training = '/scratch/s174197/data/Verse20/VertebraeLocalisation2/Verse20_training_heatmaps' #'/scratch/s174197/data/Verse20/VertebraeLocalisation2/Verse20_training_heatmaps_alldata'
img_dir_validation = '/scratch/s174197/data/Verse20/VertebraeLocalisation2/Verse20_validation_prep/img' #'/scratch/s174197/data/Verse20/VertebraeLocalisation2/Verse20_validation_prep_alldata/img'
heatmap_dir_validation = '/scratch/s174197/data/Verse20/VertebraeLocalisation2/Verse20_validation_heatmaps' #'/scratch/s174197/data/Verse20/VertebraeLocalisation2/Verse20_validation_heatmaps_alldata'

#mac
# img_dir_training = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_training_prep/img' #'/scratch/s174197/data/Verse20/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20_training_prep/img' #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data\img'
# heatmap_dir_training = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation/Verse20_training_heatmaps'
# img_dir_validation = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation/Verse20_validation_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_training_prep/img' #'/scratch/s174197/data/Verse20/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20_training_prep/img' #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data\img'
# heatmap_dir_validation = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation/Verse20_validation_heatmaps'


#Sweeping
# method
sweep_config = {
    'method': 'grid'
}

#Define paramters
parameters_dict = {
    'weight_decay': {
        'values': [1e-3, 5e-3, 1e-2, 5e-2, 1e-4, 5e-4]
    },
    'learning_rate': {
        'values': [1e-5, 1e-4, 1e-6]
    },
    'dropout': {
        'values': [0.3, 0.2, 0.1]
    }
}

#Fixed parameters!
num_epochs = 250
batch_size = 1


#Load data
VerSe_train = LoadData(img_dir=img_dir_training, heatmap_dir=heatmap_dir_training,transform=None)
train_loader = DataLoader(VerSe_train, batch_size=batch_size,
                        shuffle=True, num_workers=0) #SET TO True!
VerSe_val = LoadData(img_dir=img_dir_validation, heatmap_dir=heatmap_dir_validation)
val_loader = DataLoader(VerSe_val, batch_size=batch_size,
                        shuffle=False, num_workers=0) #SET TO False!


sweep_config['parameters'] = parameters_dict
sweep_id = wandb.sweep(sweep_config, project="New_VertebraeLocalisation2_SWEEP", entity='andreasaspe')


#Train loop
def main(config=None):
    with wandb.init(config=config):
        config = wandb.config

        #Unpack parameters
        lr = config.learning_rate
        wd = config.weight_decay
        dropout = config.dropout


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

        for epoch in range(num_epochs):

            print("Epoch nr.: "+str(epoch))
            
            #Train model
            model.train()

            #Batching through data
            for inputs, targets, subject in tqdm(train_loader): #tqdm(train_loader)

                #Send to device
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass, compute gradients, perform one training step.
                output, _, _ = model(inputs) #Output
                loss = loss_fn(output, targets) #Loss
                optimizer.zero_grad() #Clean up gradients
                loss.backward() #Compute gradients
                optimizer.step () #Clean up gradients

            #Do evaluation every 10 epoch
            if epoch%10 == 0:
                print("EVALUATION!")
                model.eval() #Set to evaluation

                #Training evaluation
                train_loss_eval = []
                with torch.no_grad():
                    for inputs, targets, subject in train_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        output, _, _ = model(inputs)
                        loss = loss_fn(output, targets)
                        # Save loss
                        train_loss_eval.append(loss.item())
                avg_loss_train = np.mean(train_loss_eval)
                print("Train loss: "+str(avg_loss_train))
                train_loss.append(avg_loss_train)

                #Validation evaluation
                val_loss_eval = []
                with torch.no_grad():
                        for inputs, targets, _ in val_loader:
                            inputs, targets = inputs.to(device), targets.to(device)
                            output, _, _ = model(inputs)
                            loss = loss_fn(output, targets)
                            # Save loss
                            val_loss_eval.append(loss.item())
                avg_loss_val = np.mean(val_loss_eval)
                print("Validation loss: "+str(avg_loss_val))
                val_loss.append(avg_loss_train)

                #Log in wandb
                wandb.log({"Train_loss": avg_loss_train, "Validation_loss": avg_loss_val})


wandb.agent(sweep_id, main, count=200)

if __name__ == "__main__":
    main()














        # print(subject[0])
        # if subject[0] == 'sub-verse824': #'sub-verse564' er lige før edge case, som er 824
        #      print("Now")

        # continue

#OLD TRAINING LOOP
# #Train loop
# for epoch in range(num_epochs):

#     print("Epoch nr.: "+str(epoch))
#     train_loss_batches = []
    
#     #Train model
#     model.train()

#     #Batching through data
#     for inputs, targets, subject in tqdm(train_loader): #tqdm(train_loader)
#         print(subject[0])

#         if not subject[0] == 'sub-verse564':
#             continue
    
#         #if subject[0] == 'sub-verse537': #Godt subject!
#         #show_slices_dim1(inputs[0,0,:,:,:],subject[0],no_slices=40)
#         #show_heatmap_dim1(targets[0,0,:,:,:], subject[0])
#         #show_heatmap_img_dim1(inputs[0,0,:,:,:], targets[0,0,:,:,:], subject[0])


#         #Send to device
#         inputs, targets = inputs.to(device), targets.to(device)

#         # Forward pass, compute gradients, perform one training step.
#         output, _, _ = model(inputs) #Output
#         loss = loss_fn(output, targets) #Loss
#         optimizer.zero_grad() #Clean up gradients
#         loss.backward() #Compute gradients
#         optimizer.step () #Clean up gradients

#         # Save loss
#         train_loss_batches.append(loss.item())


#     # #Back to epoch loop
#     avg_loss_train = np.mean(train_loss_batches)
#     train_loss.append(avg_loss_train)
#     print("          EPOCH TRAINING LOSS: "+str(avg_loss_train))


#     # # Compute accuracies on validation set.
#     model.eval() #Set to evaluation
#     val_loss_batches = []
#     with torch.no_grad():
#         for inputs, targets, _ in tqdm(val_loader):
#             inputs, targets = inputs.to(device), targets.to(device)
#             output, _, _ = model(inputs)
#             loss = loss_fn(output, targets)
#             # Save loss
#             val_loss_batches.append(loss.item())

 
#     avg_loss_validation = np.mean(val_loss_batches)
#     val_loss.append(avg_loss_validation)
#     print("          EPOCH VALIDATION LOSS: "+str(avg_loss_validation))

#     checkpoint = {
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'epoch': epoch,
#         'train_loss': train_loss,
#         'val_loss': val_loss,
#         'parameters_dict': parameters_dict
#     }

#     # checkpoint = {
#     #     'model_state_dict': model.state_dict(),
#     #     'optimizer_state_dict': optimizer.state_dict(),
#     #     'epoch': epoch,
#     #     'train_loss': train_loss    }

#     #Save checkpoint every 100 epochs
#     #if epoch%100 == 0:
#     # checkpoint = {
#     # 'model_state_dict': model.state_dict(),
#     # 'optimizer_state_dict': optimizer.state_dict(),
#     # 'epoch': epoch,
#     # 'train_loss': train_loss,
#     # 'val_loss': val_loss,
#     # }

#     #torch.save(checkpoint, os.path.join(checkpoint_dir,str(checkpoint_filename)+'_batchsize'+str(batch_size)+'_lr'+str(lr)+'_wd'+str(wd)+'.pth'))
    
#     #Log in wandb
#     wandb.log({"Train_loss": avg_loss_train, "Validation_loss": avg_loss_validation})

#     #Gem kun hver 50 epoch.
#     if epoch%50 == 0:
#         torch.save(checkpoint, os.path.join(checkpoint_dir,str(checkpoint_filename)+str(epoch)+'_batchsize'+str(batch_size)+'_lr'+str(lr)+'_wd'+str(wd)+'.pth'))

#     torch.save(checkpoint, os.path.join(checkpoint_dir,str(checkpoint_filename)+'_general.pth'))











        # print(subject[0])

        # if not subject[0] == 'sub-verse564' fuckede før.
        #     continue
    
        #if subject[0] == 'sub-verse537': #Godt subject!
        #show_slices_dim1(inputs[0,0,:,:,:],subject[0],no_slices=40)
        #show_heatmap_dim1(targets[0,0,:,:,:], subject[0])
        #show_heatmap_img_dim1(inputs[0,0,:,:,:], targets[0,0,:,:,:], subject[0])








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