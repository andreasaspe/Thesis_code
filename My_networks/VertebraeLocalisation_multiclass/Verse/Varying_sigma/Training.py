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
import math
#My own documents
from my_plotting_functions import *
from Create_dataset import LoadData
from my_data_utils import Predict, gaussian_kernel_3d
#from VertebraeLocalisationNet import *
from new_VertebraeLocalisationNet import *

#Define paramters
parameters_dict = {
    'epochs': 1000,
    'learning_rate': 0.00001, #1e-5, # 1e-8
    'weight_decay': 1e-4,
    'batch_size': 1,
    'dropout': 0.5
}


#For everything
#gpu-cluster
img_dir_training = '/scratch/s174197/data/Verse20/VertebraeLocalisation/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_training_prep/img' #'/scratch/s174197/data/Verse20/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20_training_prep/img' #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data\img'
ctd_dir_training = '/scratch/s174197/data/Verse20/VertebraeLocalisation/Verse20_training_prep/ctd'
img_dir_validation = '/scratch/s174197/data/Verse20/VertebraeLocalisation/Verse20_validation_prep/img'
ctd_dir_validation = '/scratch/s174197/data/Verse20/VertebraeLocalisation/Verse20_validation_prep/ctd'
checkpoint_dir = '/scratch/s174197/data/Checkpoints/VertebraeLocalisation' #'/Users/andreasaspe/Library/Mobile Documents/com~apple~CloudDocs/DTU/12.semester/Thesis/My_code/My_networks/Spine_Localisation/Checkpoints'
checkpoint_filename = 'With_transform' #No underscore after this

#mac
# img_dir_training = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_training_prep/img' #'/scratch/s174197/data/Verse20/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20_training_prep/img' #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data\img'
# heatmap_dir_training = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation/Verse20_training_heatmaps'
# img_dir_validation = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation/Verse20_validation_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_training_prep/img' #'/scratch/s174197/data/Verse20/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20_training_prep/img' #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data\img'
# heatmap_dir_validation = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation/Verse20_validation_heatmaps'



transform = transforms.Compose([
    transforms.RandomRotation(10)
])




#DEFINE STUFF
#Define loss function
class CustomLoss(nn.Module):
    
    def __init__(self,sigma_init):
        super(CustomLoss, self).__init__()
        self.sigma = nn.Parameter(sigma_init)

    def forward(self, output, target, sigma, alpha):
        loss_fn = nn.MSELoss()
        loss = loss_fn(output, target)
        sigma_norm = LA.vector_norm(sigma, ord=2)**2
        return loss + alpha*sigma_norm


def new_gaussian_kernel3D(origins, meshgrid_dim, gamma=1, sigma=1):
    d=3 #dimension
    x,y,z = origins
    mesh_x,mesh_y,mesh_z = meshgrid_dim
    
    x_grid, y_grid, z_grid = np.meshgrid(np.arange(mesh_x), np.arange(mesh_y),np.arange(mesh_z),indexing='ij')
    kernel = np.exp(-((x_grid - x)**2 + (y_grid - y)**2 + (z_grid - z)**2) / (2 * sigma**2))
    factor = gamma/( (2*math.pi)**(d/2)*sigma**d   )
    heatmap = factor*kernel
    return heatmap


def generate_targets(ctd_list,image,sigma):
    #Find dimensions
    _,_,dim1,dim2,dim3 = image.shape
    #Initialise target
    no_targets = 8
    targets = torch.zeros(no_targets, dim1, dim2, dim3)
    #Loop through centroids
    for ctd in ctd_list[1:]:
        if 17 <= ctd[0] <= 24:
            ctd_items = [tensor.item() for tensor in ctd]
            ctd_index = ctd_items[0]-17
            heatmap = new_gaussian_kernel3D(origins = (ctd_items[1],ctd_items[2],ctd_items[3]), meshgrid_dim = (dim1,dim2,dim3), sigma = sigma[ctd_index])
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            targets[ctd_index ,:,:,:] = torch.from_numpy(heatmap)
    #Put in one more dimension for batch size equal to 1
    targets = targets.unsqueeze(0)
    return targets





#Load data
VerSe_train = LoadData(img_dir=img_dir_training, ctd_dir=ctd_dir_training)
train_loader = DataLoader(VerSe_train, batch_size=1,
                        shuffle=True, num_workers=0) #SET TO True!
VerSe_val = LoadData(img_dir=img_dir_validation, heatmap_dir=ctd_dir_validation)
val_loader = DataLoader(VerSe_val, batch_size=1,
                        shuffle=False, num_workers=0) #SET TO False!


#Unpack parameters
num_epochs = parameters_dict['epochs']
lr = parameters_dict['learning_rate']
wd = parameters_dict['weight_decay']
batch_size = parameters_dict['batch_size']
dropout = parameters_dict['dropout']


#Start wand.db
wandb.init(
    # set the wandb project where this run will be logged
    project="VertebraeLocalisation",
    entity='andreasaspe',
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": lr,
    "epochs": num_epochs,
    'weight_decay': wd,
    'batch_size': batch_size,
    'drop_out': dropout,
    'Checkpoint_filename': checkpoint_filename,
    }
)



#Define model
model = VertebraeLocalisationNet()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #device = torch.device('cpu')
model.to(device)


#Define loss function and optimizer
sigma_init = torch.tensor([2.,2.,2.,2.,2.,2.,2.,2.,])*2
loss_fn = CustomLoss(sigma_init=sigma_init)
params_optimize = list(model.parameters())+[loss_fn.sigma]
optimizer = optim.Adam(params_optimize, lr = lr, weight_decay=wd) #Oprindeligt stod der 0.0001
# optimizer = optim.SGD(params=model.parameters(), lr = lr, weight_decay=wd, momentum=0.1, nesterov=True)


train_loss = []
val_loss = []


#Train loop
for epoch in range(num_epochs):

    print("Epoch nr.: "+str(epoch))
    train_loss_batches = []
    
    #Train model
    model.train()

    #Batching through data
    for inputs, centroids, subject in tqdm(train_loader):
    #if subject[0] == 'sub-verse537': #Godt subject!
        #Generate targets
        targets = generate_targets(centroids,inputs,loss_fn.sigma.detach().numpy())
        #show_heatmap_img_dim1(inputs[0,0,:,:,:],targets[0,0,:,:,:],subject[0])
        #Send to device
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass, compute gradients, perform one training step.
        output = model(inputs) #Output
        loss = loss_fn(output, targets, loss_fn.sigma, alpha) #Loss
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
    val_loss_batches = []
    with torch.no_grad():
        model.eval()
        for inputs, centroids, _ in tqdm(val_loader):
            #Generate targets
            targets = generate_targets(centroids,inputs,loss_fn.sigma.detach().numpy())
            #Send to device
            inputs, targets = inputs.to(device), targets.to(device)

            output = model(inputs)
            loss = loss_fn(output, targets, loss_fn.sigma, alpha) #Loss
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
        'sigma': loss_fn.sigma.detach().numpy()
    }

    #torch.save(checkpoint, os.path.join(checkpoint_dir,'checkpoint'+str(epoch+1)+'.pth'))
    torch.save(checkpoint, os.path.join(checkpoint_dir,str(checkpoint_filename)+'_batchsize'+str(batch_size)+'_lr'+str(lr)+'_wd'+str(wd)+'.pth'))

    #Log in wandb
    wandb.log({"Train_loss": avg_loss_train, "Validation_loss": avg_loss_validation, "Sigma": loss_fn.sigma.detach().numpy()})



























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