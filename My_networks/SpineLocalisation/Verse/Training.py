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
import wandb
#import albumentations as A
#My own documents
from my_plotting_functions import *
from SpineLocalisationNet import *
from Create_dataset import LoadData
from my_data_utils import Predict


#DEFINE STUFF
parameters_dict = {
    'epochs': 3000,
    'learning_rate': 1e-4,
    'weight_decay': 5e-4,
    'batch_size': 1,
    'dropout': 0.0,
    'transform': 'rotation'
}



#For everything
#Training
img_dir_training = '/scratch/s174197/data/Verse20/SpineLocalisation/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_training_prep/img' #'/scratch/s174197/data/Verse20/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20_training_prep/img' #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data\img'
heatmap_dir_training = '/scratch/s174197/data/Verse20/SpineLocalisation/Verse20_training_heatmaps' #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_training_heatmaps' #'/scratch/s174197/data/Verse20/Verse20_training_heatmaps' #'/Users/andreasaspe/Documents/Data/Verse20_training_heatmaps' #r'C:\Users\PC\Documents\Andreas_s174197\heatmaps'
#Validation
img_dir_validation = '/scratch/s174197/data/Verse20/SpineLocalisation/Verse20_validation_prep/img'
heatmap_dir_validation = '/scratch/s174197/data/Verse20/SpineLocalisation/Verse20_validation_heatmaps'
#Checkpoint and wandb
checkpoint_dir = '/scratch/s174197/data/Checkpoints/SpineLocalisation' #'/home/s174197/data/Checkpoints/SpineLocalisation' #'/Users/andreasaspe/Library/Mobile Documents/com~apple~CloudDocs/DTU/12.semester/Thesis/My_code/My_networks/Spine_Localisation/Checkpoints'
run_name = 'Only_rotation'
description = 'Prøver med data augmentation igen'
#For predition
img_dir_test = '/scratch/s174197/data/Verse20/SpineLocalisation/Verse20_test_prep/img' #'/scratch/s174197/data/Verse20/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20_training_prep/img' #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data\img'
heatmap_dir_test = '/scratch/s174197/data/Verse20/SpineLocalisation/Verse20_test_heatmaps' #'/scratch/s174197/data/Verse20/Verse20_training_heatmaps' #'/Users/andreasaspe/Documents/Data/Verse20_training_heatmaps' #r'C:\Users\PC\Documents\Andreas_s174197\heatmaps'
output_parent_dir = '/scratch/s174197/data/Verse20/SpineLocalisation/Verse20_test_heatmaps_predictions' #Prediction directory


#mac
# img_dir_training = '/Users/andreasaspe/Documents/Data/Verse20/SpineLocalisation/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_training_prep/img' #'/scratch/s174197/data/Verse20/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20_training_prep/img' #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data\img'
# heatmap_dir_training = '/Users/andreasaspe/Documents/Data/Verse20/SpineLocalisation/Verse20_training_heatmaps'
# img_dir_validation = '/Users/andreasaspe/Documents/Data/Verse20/SpineLocalisation/Verse20_validation_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_training_prep/img' #'/scratch/s174197/data/Verse20/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20_training_prep/img' #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data\img'
# heatmap_dir_validation = '/Users/andreasaspe/Documents/Data/Verse20/SpineLocalisation/Verse20_validation_heatmaps'

#Unpack parameters
num_epochs = parameters_dict['epochs']
lr = parameters_dict['learning_rate']
wd = parameters_dict['weight_decay']
batch_size = parameters_dict['batch_size']
dropout = parameters_dict['dropout']
transform = parameters_dict['transform']

#Start wand.db
wandb.init(
    # set the wandb project where this run will be logged
    project="New_SpineLocalisation",
    entity='andreasaspe',
    name=run_name,
    notes = description,
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": lr,
    "epochs": num_epochs,
    'weight_decay': wd,
    'batch_size': batch_size,
    'drop_out': dropout,
    'transform': transform,
    'run_name': run_name,
    }
)

#Define model
model = Unet3D()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #device = torch.device('cpu')
model.to(device)


#Define values
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay=wd) #Oprindeligt stod der 0.0001

#Load data
VerSe_train = LoadData(img_dir=img_dir_training,heatmap_dir=heatmap_dir_training,transform=transform)
VerSe_val = LoadData(img_dir=img_dir_validation,heatmap_dir=heatmap_dir_validation)
train_loader = DataLoader(VerSe_train, batch_size=batch_size,
                        shuffle=True, num_workers=0)
val_loader = DataLoader(VerSe_val, batch_size=batch_size,
                        shuffle=False, num_workers=0)


#Load checkpoint
# checkpoint = torch.load(os.path.join(checkpoint_dir,'checkpoint.pth'))
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


train_loss = []
val_loss = []

#Train loop
for epoch in range(num_epochs):
    
    print("Epoch nr.: "+str(epoch))
    train_loss_batches = []
    
    #Train model
    model.train()
    
    #Batching through data
    for inputs, targets, _ in tqdm(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # show_heatmap_img_dim1(inputs[0,0,:,:,:], targets[0,0,:,:,:], 'dontknow')
        # show_heatmap_img_dim2(inputs[0,0,:,:,:], targets[0,0,:,:,:], 'dontknow')
        # show_heatmap_img_dim3(inputs[0,0,:,:,:], targets[0,0,:,:,:], 'dontknow')

         # Forward pass, compute gradients, perform one training step.
        output = model(inputs) #Output
        loss = loss_fn(output, targets) #Loss
        optimizer.zero_grad() #Clean up gradients
        loss.backward() #Compute gradients
        optimizer.step () #Clean up gradients
        
        # Save loss
        train_loss_batches.append(loss.item())

        #print("     Loss after "+str(step)+" steps: "+str(loss.item()))

    #Back to epoch loop
    avg_loss_train = np.mean(train_loss_batches)
    train_loss.append(avg_loss_train)
    print("          EPOCH TRAINING LOSS: "+str(avg_loss_train))

    # Compute accuracies on training and validation set.
    model.eval() #Set to evaluation
    train_loss_batches = []
    val_loss_batches = []
    with torch.no_grad():
        for inputs, targets, _ in tqdm(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            # Save loss
            val_loss_batches.append(loss.item())

    avg_loss_validation = np.mean(val_loss_batches)
    val_loss.append(avg_loss_validation)
    print("          EPOCH VALIDATION LOSS: "+str(avg_loss_validation))

    # Save checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'parameters_dict': parameters_dict,
        'run_name': run_name,
        'transform': transform
    }

    #Log in wandb
    wandb.log({"Train_loss": avg_loss_train, "Validation_loss": avg_loss_validation})
    torch.save(checkpoint, os.path.join(checkpoint_dir,str(run_name)+'_batchsize'+str(batch_size)+'_lr'+str(lr)+'_wd'+str(wd)+'.pth'))



    #Save checkpoint every 50 epoch
    # if epoch%50 == 0:
        # torch.save(checkpoint, os.path.join(checkpoint_dir,str(checkpoint_filename)+str(epoch)+'_batchsize'+str(batch_size)+'_lr'+str(lr)+'_wd'+str(wd)+'.pth'))






    # torch.save(checkpoint, os.path.join(checkpoint_dir,str(checkpoint_filename)+'_general.pth'))
    # #torch.save(checkpoint, os.path.join(checkpoint_dir,str(checkpoint_filename)+'_batchsize'+str(batch_size)+'_lr'+str(lr)+'_wd'+str(wd)+'.pth'))
    















# #DO PREDICTIONS
# print("DOING PREDICTIONS")
# #Load data
# VerSe_test = LoadData(img_dir=img_dir_test,heatmap_dir=heatmap_dir_test)
# test_loader = DataLoader(VerSe_test, batch_size=1,
#                         shuffle=False, num_workers=0)
# model = Unet3D()

# checkpoint_file_dir = os.path.join(os.path.join(checkpoint_dir,str(checkpoint_filename)+'_batchsize'+str(batch_size)+'_lr'+str(lr)+'_wd'+str(wd)+'.pth'))
# output_pred_dir = os.path.join(output_parent_dir,str(checkpoint_filename)+'_batchsize'+str(batch_size)+'_lr'+str(lr)+'_wd'+str(wd))
# Predict(test_loader, model, checkpoint_dir, output_pred_dir)
















# # HOW TO LOAD CHECKPOINT!
# checkpoint = torch.load('checkpoint.pth')
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']

    # plt.figure()
    # plt.plot(epoch_loss)
    # plt.show()
        


# for i in range(len(VerSe_dataset)):
#     if i == 5:
#         img,heatmap = VerSe_dataset[i]
#         show_heatmap_dim1(img,heatmap)
#         #show_heatmap_dim1(img,heatmap,no_slices=40)


# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import torch
# from torch import nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader
# import torchvision
# import torchvision.transforms as transforms
# from torchvision.utils import make_grid
# from sklearn import metrics
# import torch.nn as nn



# loss_fn = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr = 0.0001, weight_decay=1e-4)

# # Test the forward pass with dummy data
# out = model(torch.randn(2, 3, 32, 32, device=device))
# print("Output shape:", out.size())
# #print(f"Output logits:\n{out.detach().cpu().numpy()}")
# #print(f"Output probabilities:\n{out.softmax(1).detach().cpu().numpy()}")

# batch_size = 64
# num_epochs = 10
# validation_every_steps = 500

# step = 0
# model.train()

# train_accuracies = []
# valid_accuracies = []
        
# for epoch in range(num_epochs):
    
#     train_accuracies_batches = []
    
#     for inputs, targets in train_loader:
#         inputs, targets = inputs.to(device), targets.to(device)
        
#         # Forward pass, compute gradients, perform one training step.
#         output = model(inputs) #Output
#         loss = loss_fn(output, targets) #Loss
#         optimizer.zero_grad() #Clean up gradients
#         loss.backward() #Compute gradients
#         optimizer.step () #Clean up gradients
        
#         # Increment step counter
#         step += 1
        
#         # Compute accuracy.
#         predictions = output.max(1)[1]
#         train_accuracies_batches.append(accuracy(targets, predictions))
        
#         if step % validation_every_steps == 0:
            
#             # Append average training accuracy to list.
#             train_accuracies.append(np.mean(train_accuracies_batches))
            
#             train_accuracies_batches = []
        
#             # Compute accuracies on validation set.
#             valid_accuracies_batches = []
#             with torch.no_grad():
#                 model.eval()
#                 for inputs, targets in test_loader:
#                     inputs, targets = inputs.to(device), targets.to(device)
#                     output = model(inputs)
#                     loss = loss_fn(output, targets)

#                     predictions = output.max(1)[1]

#                     # Multiply by len(x) because the final batch of DataLoader may be smaller (drop_last=False).
#                     valid_accuracies_batches.append(accuracy(targets, predictions) * len(inputs))

#                 model.train()
                
#             # Append average validation accuracy to list.
#             valid_accuracies.append(np.sum(valid_accuracies_batches) / len(test_set))
     
#             print(f"Step {step:<5}   training accuracy: {train_accuracies[-1]}")
#             print(f"             test accuracy: {valid_accuracies[-1]}")

# print("Finished training.")