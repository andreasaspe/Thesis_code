#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 12:59:19 2023

@author: andreasaspe
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('default')
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12  # Default text - så fx. akse-tal osv.
plt.rcParams["axes.titlesize"] = 22 # Size for titles
plt.rcParams["axes.labelsize"] = 18  # Size for labels
plt.rcParams["legend.fontsize"] = 12  # Size for legends
plt.rcParams["figure.figsize"] = (6.4, 4.8)

##### DROPOUT #####
checkpoint_parent_dir = '/Users/andreasaspe/Documents/Checkpoints/VertebraeLocalisation/To_report'

#Dropout0-5
#Filenames
filename0 = 'FIXEDVAL_dropout0_batchsize1_lr1e-05_wd1e-05.pth'
filename1 = 'FIXEDVAL_dropout1_batchsize1_lr1e-05_wd1e-05.pth'
filename2 = 'FIXEDVAL_dropout2_batchsize1_lr1e-05_wd1e-05.pth' 
filename3 = 'FIXEDVAL_dropout3_batchsize1_lr1e-05_wd1e-05.pth'
filename4 = 'FIXEDVAL_dropout4_batchsize1_lr1e-05_wd1e-05.pth'
filename5 = 'FIXEDVAL_dropout5_batchsize1_lr1e-05_wd1e-05.pth'
#Directories
checkpoint_dir0 = os.path.join(checkpoint_parent_dir,filename0)
checkpoint_dir1 = os.path.join(checkpoint_parent_dir,filename1)
checkpoint_dir2 = os.path.join(checkpoint_parent_dir,filename2)
checkpoint_dir3 = os.path.join(checkpoint_parent_dir,filename3)
checkpoint_dir4 = os.path.join(checkpoint_parent_dir,filename4)
checkpoint_dir5 = os.path.join(checkpoint_parent_dir,filename5)

#Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #device = torch.device('cpu')

#Load checkpoints
checkpoint0 = torch.load(checkpoint_dir0,map_location=device)
checkpoint1 = torch.load(checkpoint_dir1,map_location=device)
checkpoint2 = torch.load(checkpoint_dir2,map_location=device)
checkpoint3 = torch.load(checkpoint_dir3,map_location=device)
checkpoint4 = torch.load(checkpoint_dir4,map_location=device)
checkpoint5 = torch.load(checkpoint_dir5,map_location=device)
#Load validation loss
val_loss0 = checkpoint0['val_loss']
val_loss1 = checkpoint1['val_loss']
val_loss2 = checkpoint2['val_loss']
val_loss3 = checkpoint3['val_loss']
val_loss4 = checkpoint4['val_loss']
val_loss5 = checkpoint5['val_loss']
#Load train loss
train_loss0 = checkpoint0['train_loss']
train_loss1 = checkpoint1['train_loss']
train_loss2 = checkpoint2['train_loss']
train_loss3 = checkpoint3['train_loss']
train_loss4 = checkpoint4['train_loss']
train_loss5 = checkpoint5['train_loss']

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

alpha = 0.5

plt.figure()
plt.plot(val_loss0,label='0.0',color=colors[0],linewidth=2)
plt.plot(val_loss1,label='0.1',color=colors[1],linewidth=2)
plt.plot(val_loss2,label='0.2',color=colors[2],linewidth=2)
plt.plot(val_loss3,label='0.3',color=colors[3],linewidth=2)
plt.plot(val_loss4,label='0.4',color=colors[4],linewidth=2)
plt.plot(val_loss5,label='0.5',color=colors[5],linewidth=2)
plt.plot(train_loss0,color=colors[0],alpha=alpha,linewidth=0.5)
plt.plot(train_loss1,color=colors[1],alpha=alpha,linewidth=0.5)
plt.plot(train_loss2,color=colors[2],alpha=alpha,linewidth=0.5)
plt.plot(train_loss3,color=colors[3],alpha=alpha,linewidth=0.5)
plt.plot(train_loss4,color=colors[4],alpha=alpha,linewidth=0.5)
plt.plot(train_loss5,color=colors[5],alpha=alpha,linewidth=0.5)
plt.legend()
plt.xlim([0,400])
plt.ylim([0,0.0009])
plt.grid()
plt.title("Run A")
# plt.title("Different dropout ratios")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.show()





##### Initialization #####
#Filenames
filename0001 = 'init0.0001.csv'
filename001 = 'init0.001.csv'
filename01 = 'init0.01.csv' 
filename1 = 'init0.1.csv'
#Directories
file_dir0001 = os.path.join(checkpoint_parent_dir,filename0001)
file_dir001 = os.path.join(checkpoint_parent_dir,filename001)
file_dir01 = os.path.join(checkpoint_parent_dir,filename01)
file_dir1 = os.path.join(checkpoint_parent_dir,filename1)
#Load files
df0001 = pd.read_csv(file_dir0001)
df001 = pd.read_csv(file_dir001)
df01 = pd.read_csv(file_dir01)
df1 = pd.read_csv(file_dir1)
#Get val loss
val_loss0001 = df0001['true-sea-66 - Validation_loss']
val_loss001 = df001['true-night-67 - Validation_loss']
val_loss01 = df01['wild-sun-68 - Validation_loss']
val_loss1 = df1['lyric-microwave-69 - Validation_loss']
#Get train loss
train_loss0001 = df0001['true-sea-66 - Train_loss']
train_loss001 = df001['true-night-67 - Train_loss']
train_loss01 = df01['wild-sun-68 - Train_loss']
train_loss1 = df1['lyric-microwave-69 - Train_loss']

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

alpha = 0.5


plt.figure()
plt.plot(val_loss0001,label='$10^{-4}$',color=colors[0],linewidth=2)
plt.plot(val_loss001,label='$10^{-3}$',color=colors[1],linewidth=2)
plt.plot(val_loss01,label='$10^{-2}$',color=colors[2],linewidth=2)
plt.plot(val_loss1,label='$10^{-1}$',color=colors[3],linewidth=2)
plt.plot(train_loss0001,color=colors[0],linewidth=0.5)
plt.plot(train_loss001,color=colors[1],linewidth=0.5)
plt.plot(train_loss01,color=colors[2],linewidth=0.5)
plt.plot(train_loss1,color=colors[3],linewidth=0.5)
plt.legend(title=r"$\sigma$")
plt.xlim([0,900])
plt.grid()
plt.title("Run B")
# plt.title("Different standard variation for weight initialisation")
plt.ylabel("Loss")
plt.xlabel("Epochs")


##### Learning_rate #####
#Filenames
filename0 = 'TOREPORT_learningrate_LeakyReluinfirstnetworkonly_newinit_nodropout_batchsize1_lr0.0001_wd0.0001.pth'
filename1 = 'TOREPORT_learningrate_LeakyReluinfirstnetworkonly_newinit_nodropout_batchsize1_lr1e-05_wd0.0001.pth'
filename2 = 'TOREPORT_learningrate_LeakyReluinfirstnetworkonly_newinit_nodropout_batchsize1_lr1e-06_wd0.0001.pth'
filename3 = 'TOREPORT_learningrate_LeakyReluinfirstnetworkonly_newinit_nodropout_batchsize1_lr1e-07_wd0.0001.pth'
filename4 = 'TOREPORT_learningrate_LeakyReluinfirstnetworkonly_newinit_nodropout_batchsize1_lr1e-08_wd0.0001.pth'
#Directories
checkpoint_dir0 = os.path.join(checkpoint_parent_dir,filename0)
checkpoint_dir1 = os.path.join(checkpoint_parent_dir,filename1)
checkpoint_dir2 = os.path.join(checkpoint_parent_dir,filename2)
checkpoint_dir3 = os.path.join(checkpoint_parent_dir,filename3)
checkpoint_dir4 = os.path.join(checkpoint_parent_dir,filename4)

#Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #device = torch.device('cpu')

#Load checkpoints
checkpoint0 = torch.load(checkpoint_dir0,map_location=device)
checkpoint1 = torch.load(checkpoint_dir1,map_location=device)
checkpoint2 = torch.load(checkpoint_dir2,map_location=device)
checkpoint3 = torch.load(checkpoint_dir3,map_location=device)
checkpoint4 = torch.load(checkpoint_dir4,map_location=device)
#Load validation loss
val_loss0 = checkpoint0['val_loss']
val_loss1 = checkpoint1['val_loss']
val_loss2 = checkpoint2['val_loss']
val_loss3 = checkpoint3['val_loss']
val_loss4 = checkpoint4['val_loss']
#Load train loss
train_loss0 = checkpoint0['train_loss']
train_loss1 = checkpoint1['train_loss']
train_loss2 = checkpoint2['train_loss']
train_loss3 = checkpoint3['train_loss']
train_loss4 = checkpoint4['train_loss']

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

alpha = 0.5

plt.figure()
plt.plot(val_loss0,label='$10^{-4}$',color=colors[0],linewidth=2)
plt.plot(val_loss1,label='$10^{-5}$',color=colors[1],linewidth=2)
plt.plot(val_loss2,label='$10^{-6}$',color=colors[2],linewidth=2)
plt.plot(val_loss3,label='$10^{-7}$',color=colors[3],linewidth=2)
plt.plot(val_loss4,label='$10^{-8}$',color=colors[4],linewidth=2)
plt.plot(train_loss0,color=colors[0],alpha=alpha,linewidth=0.5)
plt.plot(train_loss1,color=colors[1],alpha=alpha,linewidth=0.5)
plt.plot(train_loss2,color=colors[2],alpha=alpha,linewidth=0.5)
plt.plot(train_loss3,color=colors[3],alpha=alpha,linewidth=0.5)
plt.plot(train_loss4,color=colors[4],alpha=alpha,linewidth=0.5)
plt.legend()
plt.xlim([0,1750])
plt.ylim([0,0.0015])
plt.grid()
plt.title("Run C")
# plt.title("Different learning rates")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.show()



##### Weight decay #####
#Filenames
filename0 = 'TOREPORT_weightdecay_LeakyReluinfirstnetworkonly_newinit_nodropout_batchsize1_lr1e-05_wd0.001.pth'
filename1 = 'TOREPORT_weightdecay_LeakyReluinfirstnetworkonly_newinit_nodropout_batchsize1_lr1e-05_wd0.0001.pth'
filename2 = 'TOREPORT_weightdecay_LeakyReluinfirstnetworkonly_newinit_nodropout_batchsize1_lr1e-05_wd1e-05.pth'
filename3 = 'TOREPORT_weightdecay_LeakyReluinfirstnetworkonly_newinit_nodropout_batchsize1_lr1e-05_wd1e-06.pth'
#Directories
checkpoint_dir0 = os.path.join(checkpoint_parent_dir,filename0)
checkpoint_dir1 = os.path.join(checkpoint_parent_dir,filename1)
checkpoint_dir2 = os.path.join(checkpoint_parent_dir,filename2)
checkpoint_dir3 = os.path.join(checkpoint_parent_dir,filename3)

#Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #device = torch.device('cpu')

#Load checkpoints
checkpoint0 = torch.load(checkpoint_dir0,map_location=device)
checkpoint1 = torch.load(checkpoint_dir1,map_location=device)
checkpoint2 = torch.load(checkpoint_dir2,map_location=device)
checkpoint3 = torch.load(checkpoint_dir3,map_location=device)
#Load validation loss
val_loss0 = checkpoint0['val_loss']
val_loss1 = checkpoint1['val_loss']
val_loss2 = checkpoint2['val_loss']
val_loss3 = checkpoint3['val_loss']
#Load train loss
train_loss0 = checkpoint0['train_loss']
train_loss1 = checkpoint1['train_loss']
train_loss2 = checkpoint2['train_loss']
train_loss3 = checkpoint3['train_loss']

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

alpha = 0.5

plt.figure()
plt.plot(val_loss0,label='$10^{-3}$',color=colors[0],linewidth=2)
plt.plot(val_loss1,label='$10^{-4}$',color=colors[1],linewidth=2)
plt.plot(val_loss2,label='$10^{-5}$',color=colors[2],linewidth=2)
plt.plot(val_loss3,label='$10^{-6}$',color=colors[3],linewidth=2)
plt.plot(train_loss0,color=colors[0],alpha=alpha,linewidth=0.5)
plt.plot(train_loss1,color=colors[1],alpha=alpha,linewidth=0.5)
plt.plot(train_loss2,color=colors[2],alpha=alpha,linewidth=0.5)
plt.plot(train_loss3,color=colors[3],alpha=alpha,linewidth=0.5)
plt.legend()
plt.xlim([0,1000])
plt.ylim([0,0.001])
plt.grid()
plt.title("Run D")
# plt.title("Different values of weight decay")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.show()



##### Plot to report #####
#Filenames
filename0 = 'data_augmentation_batchsize1_lr1e-05_wd0.0001.pth' #Det her checkpoint indeholder ikke noget training-loss data: 'NEWNetworkWithNewInitialisation_GPU_batchsize1_lr0.001_wd0.pth' 
#Directories
checkpoint_dir0 = os.path.join(checkpoint_parent_dir,filename0)

#Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #device = torch.device('cpu')

#Load checkpoints
checkpoint0 = torch.load(checkpoint_dir0,map_location=device)
#Load validation loss
val_loss0 = checkpoint0['val_loss']
#Load train loss
train_loss0 = checkpoint0['train_loss']

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

alpha = 0.5

plt.figure()
plt.plot(val_loss0,label='Validation loss',color=colors[1],linewidth=2)
plt.plot(train_loss0,label='Training loss',color=colors[0],linewidth=2)
plt.legend(fontsize=15)
plt.xlim([0,600])
plt.ylim([0,0.0006])
plt.grid()
# plt.title("Validation loss for different values of weight decay")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.show()

