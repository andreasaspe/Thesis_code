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

# ##### Weight decay #####
checkpoint_parent_dir = '/Users/andreasaspe/Documents/Checkpoints/VertebraeLocalisation2/To_report'

# #Dropout0-5
# #Filenames
filename5 = 'Weight_decay5e-3_batchsize1_lr1e-05_wd0.005.pth'
filename0 = 'Weight_decay1e-1_batchsize1_lr1e-05_wd0.1.pth'
filename3 = 'Weight_decay5e-2_batchsize1_lr1e-05_wd0.05.pth'
filename7 = 'Weight_decay5e-4_batchsize1_lr1e-05_wd0.0005.pth'
filename2 = 'Weight_decay1e-2_batchsize1_lr1e-05_wd0.01.pth'
filename6 = 'Weight_decay1e-4_batchsize1_lr1e-05_wd0.0001.pth'
filename1 = 'Weight_decay5e-1_batchsize1_lr1e-05_wd0.5.pth'
filename4 = 'Weight_decay1e-3_batchsize1_lr1e-05_wd0.001.pth'
# Filenames
run_name5 = 'Weight_decay5e-3'
run_name0 = 'Weight_decay1e-1'
run_name3 = 'Weight_decay5e-2'
run_name7 = 'Weight_decay5e-4'
run_name2 = 'Weight_decay1e-2'
run_name6 = 'Weight_decay1e-4'
run_name1 = 'Weight_decay5e-1'
run_name4 = 'Weight_decay1e-3'
#To csv.
filename0 = run_name0+'.csv'
filename1 = run_name1+'.csv'
filename2 = run_name2+'.csv'
filename3 = run_name3+'.csv'
filename4 = run_name4+'.csv'
filename5 = run_name5+'.csv'
filename6 = run_name6+'.csv'
filename7 = run_name7+'.csv'

#Directories
file_dir0 = os.path.join(checkpoint_parent_dir,filename0)
file_dir1 = os.path.join(checkpoint_parent_dir,filename1)
file_dir2 = os.path.join(checkpoint_parent_dir,filename2)
file_dir3 = os.path.join(checkpoint_parent_dir,filename3)
file_dir4 = os.path.join(checkpoint_parent_dir,filename4)
file_dir5 = os.path.join(checkpoint_parent_dir,filename5)
file_dir6 = os.path.join(checkpoint_parent_dir,filename6)
file_dir7 = os.path.join(checkpoint_parent_dir,filename7)



#Load files
df0 = pd.read_csv(file_dir0)
df1 = pd.read_csv(file_dir1)
df2 = pd.read_csv(file_dir2)
df3 = pd.read_csv(file_dir3)
df4 = pd.read_csv(file_dir4)
df5 = pd.read_csv(file_dir5)
df6 = pd.read_csv(file_dir6)
df7 = pd.read_csv(file_dir7)


#Get val loss
val_loss0 = df0[run_name0+' - Validation_loss']
val_loss1 = df1[run_name1+' - Validation_loss']
val_loss2 = df2[run_name2+' - Validation_loss']
val_loss3 = df3[run_name3+' - Validation_loss']
val_loss4 = df4[run_name4+' - Validation_loss']
val_loss5 = df5[run_name5+' - Validation_loss']
val_loss6 = df6[run_name6+' - Validation_loss']
val_loss7 = df7[run_name7+' - Validation_loss']

#Get train loss++++
train_loss0 = df0[run_name0+' - Train_loss']
train_loss1 = df1[run_name1+' - Train_loss']
train_loss2 = df2[run_name2+' - Train_loss']
train_loss3 = df3[run_name3+' - Train_loss']
train_loss4 = df4[run_name4+' - Train_loss']
train_loss5 = df5[run_name5+' - Train_loss']
train_loss6 = df6[run_name6+' - Train_loss']
train_loss7 = df7[run_name7+' - Train_loss']



colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',  '#17becf']

alpha = 0.5

# fig, ax = plt.subplots(ncols=1, figsize=(6.4,4.8))
plt.figure()
plt.plot(np.arange(0,len(val_loss0)*10,10),val_loss0,label='$10^{-1}$',color=colors[0],linewidth=2)
plt.plot(np.arange(0,len(val_loss1)*10,10),val_loss1,label='$5 x 10^{-1}$',color=colors[1],linewidth=2)
plt.plot(np.arange(0,len(val_loss2)*10,10),val_loss2,label='$10^{-2}$',color=colors[2],linewidth=2)
plt.plot(np.arange(0,len(val_loss3)*10,10),val_loss3,label='$5 x 10^{-2}$',color=colors[3],linewidth=2)
plt.plot(np.arange(0,len(val_loss4)*10,10),val_loss4,label='$10^{-3}$',color=colors[4],linewidth=2)
plt.plot(np.arange(0,len(val_loss5)*10,10),val_loss5,label='$5 x 10^{-3}$',color=colors[5],linewidth=2)
plt.plot(np.arange(0,len(val_loss6)*10,10),val_loss6,label='$10^{-4}$',color=colors[6],linewidth=2)
plt.plot(np.arange(0,len(val_loss7)*10,10),val_loss7,label='$5 x 10^{-4}$',color=colors[7],linewidth=2)
plt.plot(np.arange(0,len(train_loss0)*10,10),train_loss0,color=colors[0],alpha=alpha,linewidth=0.5)
plt.plot(np.arange(0,len(train_loss1)*10,10),train_loss1,color=colors[1],alpha=alpha,linewidth=0.5)
plt.plot(np.arange(0,len(train_loss2)*10,10),train_loss2,color=colors[2],alpha=alpha,linewidth=0.5)
plt.plot(np.arange(0,len(train_loss3)*10,10),train_loss3,color=colors[3],alpha=alpha,linewidth=0.5)
plt.plot(np.arange(0,len(train_loss4)*10,10),train_loss4,color=colors[4],alpha=alpha,linewidth=0.5)
plt.plot(np.arange(0,len(train_loss5)*10,10),train_loss5,color=colors[5],alpha=alpha,linewidth=0.5)
plt.plot(np.arange(0,len(train_loss6)*10,10),train_loss6,color=colors[6],alpha=alpha,linewidth=0.5)
plt.plot(np.arange(0,len(train_loss7)*10,10),train_loss7,color=colors[7],alpha=alpha,linewidth=0.5)
plt.legend(loc = 'upper right')
plt.xlim([0,2000])
# plt.ylim([0,0.0009])
plt.grid()
plt.title("Run C")
# plt.title("Different dropout ratios")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.show()







# ##### Drop out #####
checkpoint_parent_dir = '/Users/andreasaspe/Documents/Checkpoints/VertebraeLocalisation2/To_report'

# Filenames
run_name0 = 'Dropout0'
run_name1 = 'Dropout1'
run_name2 = 'Dropout2'
run_name3 = 'Dropout3'
run_name4 = 'Dropout4'
run_name5 = 'Dropout5'
#To csv.
filename0 = run_name0+'.csv'
filename1 = run_name1+'.csv'
filename2 = run_name2+'.csv'
filename3 = run_name3+'.csv'
filename4 = run_name4+'.csv'
filename5 = run_name5+'.csv'

#Directories
file_dir0 = os.path.join(checkpoint_parent_dir,filename0)
file_dir1 = os.path.join(checkpoint_parent_dir,filename1)
file_dir2 = os.path.join(checkpoint_parent_dir,filename2)
file_dir3 = os.path.join(checkpoint_parent_dir,filename3)
file_dir4 = os.path.join(checkpoint_parent_dir,filename4)
file_dir5 = os.path.join(checkpoint_parent_dir,filename5)

#Load files
df0 = pd.read_csv(file_dir0)
df1 = pd.read_csv(file_dir1)
df2 = pd.read_csv(file_dir2)
df3 = pd.read_csv(file_dir3)
df4 = pd.read_csv(file_dir4)
df5 = pd.read_csv(file_dir5)

#Get val loss
val_loss0 = df0[run_name0+' - Validation_loss']
val_loss1 = df1[run_name1+' - Validation_loss']
val_loss2 = df2[run_name2+' - Validation_loss']
val_loss3 = df3[run_name3+' - Validation_loss']
val_loss4 = df4[run_name4+' - Validation_loss']
val_loss5 = df5[run_name5+' - Validation_loss']

#Get train loss++++
train_loss0 = df0[run_name0+' - Train_loss']
train_loss1 = df1[run_name1+' - Train_loss']
train_loss2 = df2[run_name2+' - Train_loss']
train_loss3 = df3[run_name3+' - Train_loss']
train_loss4 = df4[run_name4+' - Train_loss']
train_loss5 = df5[run_name5+' - Train_loss']



colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',  '#17becf']

alpha = 0.5

plt.figure()
plt.plot(np.arange(0,len(val_loss0)*10,10),val_loss0,label='$0.0$',color=colors[0],linewidth=2)
plt.plot(np.arange(0,len(val_loss1)*10,10),val_loss1,label='$0.1$',color=colors[1],linewidth=2)
plt.plot(np.arange(0,len(val_loss2)*10,10),val_loss2,label='$0.2$',color=colors[2],linewidth=2)
plt.plot(np.arange(0,len(val_loss3)*10,10),val_loss3,label='$0.3$',color=colors[3],linewidth=2)
plt.plot(np.arange(0,len(val_loss4)*10,10),val_loss4,label='$0.4$',color=colors[4],linewidth=2)
plt.plot(np.arange(0,len(val_loss5)*10,10),val_loss5,label='$0.5$',color=colors[5],linewidth=2)
plt.plot(np.arange(0,len(train_loss0)*10,10),train_loss0,color=colors[0],alpha=alpha,linewidth=0.5)
plt.plot(np.arange(0,len(train_loss1)*10,10),train_loss1,color=colors[1],alpha=alpha,linewidth=0.5)
plt.plot(np.arange(0,len(train_loss2)*10,10),train_loss2,color=colors[2],alpha=alpha,linewidth=0.5)
plt.plot(np.arange(0,len(train_loss3)*10,10),train_loss3,color=colors[3],alpha=alpha,linewidth=0.5)
plt.plot(np.arange(0,len(train_loss4)*10,10),train_loss4,color=colors[4],alpha=alpha,linewidth=0.5)
plt.plot(np.arange(0,len(train_loss5)*10,10),train_loss5,color=colors[5],alpha=alpha,linewidth=0.5)
plt.legend()
plt.xlim([0,800])
# plt.ylim([0,0.0009])
plt.grid()
plt.title("Run A")
# plt.title("Different dropout ratios")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.show()



# ##### Drop out alternative #####
checkpoint_parent_dir = '/Users/andreasaspe/Documents/Checkpoints/VertebraeLocalisation2/To_report'

# Filenames
filename0 = 'Dropout0_alternative_batchsize1_lr1e-05_wd0.0001.pth'
filename1 = 'Dropout1_alternative_batchsize1_lr1e-05_wd0.0001.pth'
filename2 = 'Dropout2_alternative_batchsize1_lr1e-05_wd0.0001.pth'
filename3 = 'Dropout3_alternative_batchsize1_lr1e-05_wd0.0001.pth'
filename4 = 'Dropout4_alternative_batchsize1_lr1e-05_wd0.0001.pth'
filename5 = 'Dropout5_alternative_batchsize1_lr1e-05_wd0.0001.pth'

run_name0 = 'Dropout0_alternative'
run_name1 = 'Dropout1_alternative'
run_name2 = 'Dropout2_alternative'
run_name3 = 'Dropout3_alternative'
run_name4 = 'Dropout4_alternative'
run_name5 = 'Dropout5_alternative'
#To csv.
filename0 = run_name0+'.csv'
filename1 = run_name1+'.csv'
filename2 = run_name2+'.csv'
filename3 = run_name3+'.csv'
filename4 = run_name4+'.csv'
filename5 = run_name5+'.csv'

#Directories
file_dir0 = os.path.join(checkpoint_parent_dir,filename0)
file_dir1 = os.path.join(checkpoint_parent_dir,filename1)
file_dir2 = os.path.join(checkpoint_parent_dir,filename2)
file_dir3 = os.path.join(checkpoint_parent_dir,filename3)
file_dir4 = os.path.join(checkpoint_parent_dir,filename4)
file_dir5 = os.path.join(checkpoint_parent_dir,filename5)

#Load files
df0 = pd.read_csv(file_dir0)
df1 = pd.read_csv(file_dir1)
df2 = pd.read_csv(file_dir2)
df3 = pd.read_csv(file_dir3)
df4 = pd.read_csv(file_dir4)
df5 = pd.read_csv(file_dir5)

#Get val loss
val_loss0 = df0[run_name0+' - Validation_loss']
val_loss1 = df1[run_name1+' - Validation_loss']
val_loss2 = df2[run_name2+' - Validation_loss']
val_loss3 = df3[run_name3+' - Validation_loss']
val_loss4 = df4[run_name4+' - Validation_loss']
val_loss5 = df5[run_name5+' - Validation_loss']

#Get train loss++++
train_loss0 = df0[run_name0+' - Train_loss']
train_loss1 = df1[run_name1+' - Train_loss']
train_loss2 = df2[run_name2+' - Train_loss']
train_loss3 = df3[run_name3+' - Train_loss']
train_loss4 = df4[run_name4+' - Train_loss']
train_loss5 = df5[run_name5+' - Train_loss']



colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',  '#17becf']

alpha = 0.5

plt.figure()
plt.plot(np.arange(0,len(val_loss0)*10,10),val_loss0,label='$0.0$',color=colors[0],linewidth=2)
plt.plot(np.arange(0,len(val_loss1)*10,10),val_loss1,label='$0.1$',color=colors[1],linewidth=2)
plt.plot(np.arange(0,len(val_loss2)*10,10),val_loss2,label='$0.2$',color=colors[2],linewidth=2)
plt.plot(np.arange(0,len(val_loss3)*10,10),val_loss3,label='$0.3$',color=colors[3],linewidth=2)
plt.plot(np.arange(0,len(val_loss4)*10,10),val_loss4,label='$0.4$',color=colors[4],linewidth=2)
plt.plot(np.arange(0,len(val_loss5)*10,10),val_loss5,label='$0.5$',color=colors[5],linewidth=2)
plt.plot(np.arange(0,len(train_loss0)*10,10),train_loss0,color=colors[0],alpha=alpha,linewidth=0.5)
plt.plot(np.arange(0,len(train_loss1)*10,10),train_loss1,color=colors[1],alpha=alpha,linewidth=0.5)
plt.plot(np.arange(0,len(train_loss2)*10,10),train_loss2,color=colors[2],alpha=alpha,linewidth=0.5)
plt.plot(np.arange(0,len(train_loss3)*10,10),train_loss3,color=colors[3],alpha=alpha,linewidth=0.5)
plt.plot(np.arange(0,len(train_loss4)*10,10),train_loss4,color=colors[4],alpha=alpha,linewidth=0.5)
plt.plot(np.arange(0,len(train_loss5)*10,10),train_loss5,color=colors[5],alpha=alpha,linewidth=0.5)
plt.legend()
plt.xlim([0,800])
# plt.ylim([0,0.0009])
plt.grid()
plt.title("Run B")
# plt.title("Different dropout ratios")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.show()


# ##### More hyperparameter tuning #####
# # checkpoint_parent_dir = '/Users/andreasaspe/Documents/Checkpoints/VertebraeLocalisation2/To_report'

# # # #Dropout0-5
# # #Filenames
# # filename0 = 'sweep_rotation_dropoutalternative_batchsize1_lr0.0001_wd0.0001.pth'
# # filename1 = 'sweep_rotation_batchsize1_lr0.0001_wd0.0001.pth'
# # filename2 = 'sweep_elastic_batchsize1_lr0.0001_wd0.0001.pth'
# # filename3 = 'Only_rotation_earlystopping_epoch1790_batchsize1_lr1e-05_wd0.0001.pth'
# # filename4 = 'Only_elastic_earlystopping_epoch1780_batchsize1_lr1e-05_wd0.0001.pth'
# # filename5 = 'both_elastic_and_rotation_batchsize1_lr1e-05_wd0.0001.pth'
# # filename6 = 'Batchnorm_dropout_batchsize1_lr1e-05_wd0.0001.pth'
# # filename7 = 'Recreate5_stateofart_batchsize1_lr1e-05_wd0.0001.pth'
# # filename8 = 'no_tanh_batchsize1_lr1e-05_wd0.0001.pth'

# # filenames = [filename0,filename1,filename2,filename3,filename4,filename5,filename6,filename7,filename8]

# # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #device = torch.device('cpu')

# # for filename in filenames:
# #     checkpoint_dir = os.path.join(checkpoint_parent_dir,filename)
# #     checkpoint = torch.load(checkpoint_dir,map_location=device)
# #     val_loss = checkpoint['val_loss']
# #     train_loss = checkpoint['train_loss']
    
# #     colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# #     alpha = 0.5

# #     plt.figure()
# #     plt.plot(val_loss,label='Validation loss',color=colors[1],linewidth=4)
# #     plt.plot(train_loss,label='Training loss',color=colors[0],linewidth=2)
# #     plt.legend()
# #     plt.xlim([0,600])
# #     plt.ylim([0,0.0006])
# #     plt.grid()
# #     # plt.title("Validation loss for different values of weight decay")
# #     plt.ylabel("Loss",fontsize = 12)
# #     plt.xlabel("Epochs",fontsize = 12)



##### To actual report #####
checkpoint_parent_dir = '/Users/andreasaspe/Documents/Checkpoints/VertebraeLocalisation2/To_report'


#Filenames
run_name0 = 'Batchnorm_dropout'
run_name1 = 'Dropout3'
run_name2 = 'Recreate5_stateofart'
run_name3 = 'Only_rotation'
run_name4 = 'Only_elastic_earlystopping'
#To csv.
filename0 = run_name0+'.csv'
filename1 = run_name1+'.csv'
filename2 = run_name2+'.csv'
filename3 = run_name3+'.csv'
filename4 = run_name4+'.csv'
#Directories
file_dir0 = os.path.join(checkpoint_parent_dir,filename0)
file_dir1 = os.path.join(checkpoint_parent_dir,filename1)
file_dir2 = os.path.join(checkpoint_parent_dir,filename2)
file_dir3 = os.path.join(checkpoint_parent_dir,filename3)
file_dir4 = os.path.join(checkpoint_parent_dir,filename4)
#Load files
df0 = pd.read_csv(file_dir0)
df1 = pd.read_csv(file_dir1)
df2 = pd.read_csv(file_dir2)
df3 = pd.read_csv(file_dir3)
df4 = pd.read_csv(file_dir4)
#Get val loss
val_loss0 = df0[run_name0+' - Validation_loss']
val_loss1 = df1[run_name1+' - Validation_loss']
val_loss2 = df2[run_name2+' - Validation_loss']
val_loss3 = df3[run_name3+' - Validation_loss']
val_loss4 = df4[run_name4+' - Validation_loss']

#Get train loss
train_loss0 = df0[run_name0+' - Train_loss']
train_loss1 = df1[run_name1+' - Train_loss']
train_loss2 = df2[run_name2+' - Train_loss']
train_loss3 = df3[run_name3+' - Train_loss']
train_loss4 = df4[run_name4+' - Train_loss']

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',  '#17becf']

alpha = 0.5

x_axis = np.arange(0,len(val_loss0)*10,10)

plt.figure()
plt.plot(np.arange(0,len(val_loss2)*10,10),val_loss2,label='Nothing',color=colors[0],linewidth=2)
plt.plot(np.arange(0,len(val_loss1)*10,10),val_loss1,label='Dropout',color=colors[1],linewidth=2)
# plt.plot(np.arange(0,len(val_loss0)*10,10),val_loss0,label='Batchnorm + dropout',color=colors[0],linewidth=2)
plt.plot(np.arange(0,len(val_loss3)*10,10),val_loss3,label='Dropout + Batchnorm',color=colors[3],linewidth=2)
# plt.plot(np.arange(0,len(val_loss4)*10,10),val_loss4,label='Batchnorm + dropout + elastic',color=colors[3],linewidth=2)
plt.legend(fontsize=15)
plt.xlim([0,1050])
plt.ylim([0,0.0004])
plt.grid()
plt.title("Validation loss")
# plt.title("Different dropout ratios")
plt.ylabel("MSE loss")
plt.xlabel("Epochs")
plt.show()
plt.figure()
plt.plot(np.arange(0,len(train_loss2)*10,10),train_loss2,label='Nothing',color=colors[0],linewidth=2)
plt.plot(np.arange(0,len(train_loss1)*10,10),train_loss1,label='Dropout',color=colors[1],linewidth=2)
# plt.plot(np.arange(0,len(train_loss0)*10,10),train_loss0,label='Batchnorm + dropout',color=colors[0],linewidth=2)
plt.plot(np.arange(0,len(train_loss3)*10,10),train_loss3,label='Dropout + Batchnorm',color=colors[3],linewidth=2)
# plt.plot(np.arange(0,len(train_loss4)*10,10),train_loss4,label='Batchnorm + dropout + elastic',color=colors[3],linewidth=2)
plt.legend(fontsize=15)
plt.xlim([0,1050])
plt.ylim([0,0.0004])
plt.grid()
plt.title("Training loss")
# plt.title("Different dropout ratios")
plt.ylabel("MSE loss")
plt.xlabel("Epochs")
plt.show()



#Compare data augmentation
checkpoint_parent_dir = '/Users/andreasaspe/Documents/Checkpoints/VertebraeLocalisation2/To_report'

run_name0 = 'Batchnorm_dropout'
run_name1 = 'Only_rotation'
run_name2 = 'Only_elastic'
run_name3 = 'both_elastic_and_rotation'

#To csv.
filename0 = run_name0+'.csv'
filename1 = run_name1+'.csv'
filename2 = run_name2+'.csv'
filename3 = run_name3+'.csv'


#Directories
file_dir0 = os.path.join(checkpoint_parent_dir,filename0)
file_dir1 = os.path.join(checkpoint_parent_dir,filename1)
file_dir2 = os.path.join(checkpoint_parent_dir,filename2)
file_dir3 = os.path.join(checkpoint_parent_dir,filename3)
#Load files
df0 = pd.read_csv(file_dir0)
df1 = pd.read_csv(file_dir1)
df2 = pd.read_csv(file_dir2)
df3 = pd.read_csv(file_dir3)
df4 = pd.read_csv(file_dir4)
#Get val loss
val_loss0 = df0[run_name0+' - Validation_loss']
val_loss1 = df1[run_name1+' - Validation_loss']
val_loss2 = df2[run_name2+' - Validation_loss']
val_loss3 = df3[run_name3+' - Validation_loss']

#Get train loss
train_loss0 = df0[run_name0+' - Train_loss']
train_loss1 = df1[run_name1+' - Train_loss']
train_loss2 = df2[run_name2+' - Train_loss']
train_loss3 = df3[run_name3+' - Train_loss']

plt.figure()
plt.plot(np.arange(0,len(val_loss0)*10,10),val_loss0,label='No augmentation',color=colors[0],linewidth=2)
plt.plot(np.arange(0,len(val_loss1)*10,10),val_loss1,label='Rotation',color=colors[1],linewidth=2)
plt.plot(np.arange(0,len(val_loss2)*10,10),val_loss2,label='Elastic deformation',color=colors[2],linewidth=2)
plt.plot(np.arange(0,len(val_loss3)*10,10),val_loss3,label='Both',color=colors[3],linewidth=2)
# plt.plot(np.arange(0,len(val_loss4)*10,10),val_loss4,label='Batchnorm + dropout + elastic',color=colors[3],linewidth=2)
plt.legend(fontsize=15)
plt.xlim([0,1050])
plt.ylim([0,0.0004])
plt.grid()
plt.title("Validation loss")
# plt.title("Different dropout ratios")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.show()
plt.figure()
plt.plot(np.arange(0,len(train_loss0)*10,10),train_loss0,label='No augmentation',color=colors[0],linewidth=2)
plt.plot(np.arange(0,len(train_loss1)*10,10),train_loss1,label='Rotation',color=colors[1],linewidth=2)
plt.plot(np.arange(0,len(train_loss2)*10,10),train_loss2,label='Elastic deformation',color=colors[2],linewidth=2)
plt.plot(np.arange(0,len(train_loss3)*10,10),train_loss3,label='Both',color=colors[3],linewidth=2)
# plt.plot(np.arange(0,len(train_loss4)*10,10),train_loss4,label='Batchnorm + dropout + elastic',color=colors[3],linewidth=2)
plt.legend(fontsize=15)
plt.xlim([0,1050])
plt.ylim([0,0.0004])
plt.grid()
plt.title("Training loss")
# plt.title("Different dropout ratios")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.show()


