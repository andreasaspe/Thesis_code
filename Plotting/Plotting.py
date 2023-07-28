#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 16:36:17 2023

@author: andreasaspe
"""

from my_plotting_functions import *
import nibabel as nib
import os
import torch
from os import listdir
import numpy as np

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)


#################### HEATMAP GT SpineLocalisationNet ####################
subject = 'sub-verse510'
img_dir = os.path.join('/Users/andreasaspe/Documents/Data/Verse20/Verse20_training_prep/img',subject+'_img.nii.gz')
heatmap_dir = os.path.join('/Users/andreasaspe/Documents/Data/Verse20/Verse20_training_heatmaps',subject+'_heatmap.nii.gz')

img_nib = nib.load(img_dir)
heatmap_nib = nib.load(heatmap_dir)
img_data = img_nib.get_fdata()
heatmap_data = heatmap_nib.get_fdata()

# Create a figure with 3 subplots in one row
fig, ax = plt.subplots(ncols=3, figsize=(8,20))

plt.style.use('default')

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

for col in ax:
    col.axis('off')
    
fig.subplots_adjust(wspace=0.1)

        
# Plot each subplot
i=31
ax[0].imshow(img_data[i,:,:].T,cmap="gray",origin="lower")
ax[0].imshow(heatmap_data[i,:,:].T, cmap='hot',origin="lower", vmin=heatmap_data.min(), vmax=heatmap_data.max(),alpha = 1.0*(heatmap_data[i,:,:].T>0.4)) #Jo større tal ved alpha, jo bredere kurve  ax.set_title('dim, '+str(subject)+", Slice: "+str(i))
ax[0].set_title("Sagittal")
i=26 #(eller 25)
ax[1].imshow(img_data[:,i,:].T,cmap="gray",origin="lower")
ax[1].imshow(heatmap_data[:,i,:].T, cmap='hot',origin="lower", vmin=heatmap_data.min(), vmax=heatmap_data.max(),alpha = 1.0*(heatmap_data[:,i,:].T>0.4)) #Jo større tal ved alpha, jo bredere kurve  ax.set_title('dim, '+str(subject)+", Slice: "+str(i))
ax[1].set_title("Coronal")
i=91
ax[2].imshow(img_data[:,:,i].T,cmap="gray",origin="lower")
ax[2].imshow(heatmap_data[:,:,i].T, cmap='hot',origin="lower", vmin=heatmap_data.min(), vmax=heatmap_data.max(),alpha = 1.0*(heatmap_data[:,:,i].T>0.4)) #Jo større tal ved alpha, jo bredere kurve  ax.set_title('dim, '+str(subject)+", Slice: "+str(i))
ax[2].set_title("Axial")
plt.savefig('2D_Heatmap',bbox_inches='tight')
plt.show()



#################### HEATMAP PREDICTIONS EXAMPLE SpineLocalisationNet ####################
# list_of_subjects = ['sub-verse509','sub-verse512'] #List of subjects, 'sub-verse510'512
# list_of_slices = [33,27] #List of subjects, 'sub-verse510'512
# img_dir = '/Users/andreasaspe/Documents/Data/Verse20/Verse20_test_prep/img'
# heatmap_GT_dir = '/Users/andreasaspe/Documents/Data/Verse20/Verse20_test_heatmaps'
# heatmap_pred_dir = '/Users/andreasaspe/Documents/Data/Verse20/Verse20_test_predictions'

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #device = torch.device('cpu')

# fig, ax = plt.subplots(ncols=3, nrows = len(list_of_subjects), figsize=(8,4))
# fig.subplots_adjust(wspace=-0.8,hspace=0.05)

# for row in ax:
#     for col in row:
#         col.axis('off')
        
# plt.style.use('default')

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "Helvetica"
# })

# for i, subject in enumerate(list_of_subjects):
#     filename_heatmap_pred = [f for f in listdir(heatmap_pred_dir) if f.startswith(subject)][0]
#     filename_heatmap_GT = [f for f in listdir(heatmap_GT_dir) if f.startswith(subject)][0]

#     heatmap_pred_file_dir = os.path.join(heatmap_pred_dir, filename_heatmap_pred)
#     heatmap_GT_file_dir = os.path.join(heatmap_GT_dir, filename_heatmap_GT)

#     #IMPORT PREDICTIONS
#     heatmap_data_pred = torch.load(heatmap_pred_file_dir, map_location=device)
#     heatmap_data_pred = heatmap_data_pred.detach().numpy()
#     #Normalize
#     heatmap_data_pred = (heatmap_data_pred - heatmap_data_pred.min()) / (heatmap_data_pred.max() - heatmap_data_pred.min())
    
#     #Import GT
#     heatmap_data_GT = nib.load(heatmap_GT_file_dir)
#     heatmap_data_GT = np.asanyarray(heatmap_data_GT.dataobj, dtype=np.float32)
#     #Normalize
#     heatmap_data_GT = (heatmap_data_GT - heatmap_data_GT.min()) / (heatmap_data_GT.max() - heatmap_data_GT.min())
    
#     #IMPORT IMAGE
#     filename_img = subject + "_img.nii.gz"
#     img_nib = nib.load(os.path.join(img_dir,filename_img))
#     img_data = np.asanyarray(img_nib.dataobj, dtype=np.float32)
    
#     # plt.figure()
#     # plt.imshow(img_data[list_of_slices[i],:,:].T,cmap="gray",origin="lower")
#     # plt.axis('off')
#     # plt.savefig("SpineLocalisationNet"+str(i) + "scan")


#     # plt.figure()
#     # plt.imshow(img_data[list_of_slices[i],:,:].T,cmap="gray",origin="lower")
#     # plt.imshow(heatmap_data_GT[list_of_slices[i],:,:].T, cmap='hot',origin="lower", vmin=0, vmax=1, alpha = 1.0*(heatmap_data_GT[list_of_slices[i],:,:].T>0.4)) #Jo større tal ved alpha, jo bredere kurve  ax.set_title('dim, '+str(subject)+", Slice: "+str(i))
#     # plt.axis('off')
#     # plt.savefig("SpineLocalisationNet"+str(i) + "GT")



#     # plt.figure()
#     # plt.imshow(img_data[list_of_slices[i],:,:].T,cmap="gray",origin="lower")
#     # plt.imshow(heatmap_data_pred[list_of_slices[i],:,:].T, cmap='hot',origin="lower", vmin=0, vmax=1, alpha = 1.0*(heatmap_data_pred[list_of_slices[i],:,:].T>0.4)) #Jo større tal ved alpha, jo bredere kurve  ax.set_title('dim, '+str(subject)+", Slice: "+str(i))
#     # plt.axis('off')
#     # plt.savefig("SpineLocalisationNet"+str(i) + "pred")



#     ax[i,0].imshow(img_data[list_of_slices[i],:,:].T,cmap="gray",origin="lower")

#     ax[i,1].imshow(img_data[list_of_slices[i],:,:].T,cmap="gray",origin="lower")
#     ax[i,1].imshow(heatmap_data_GT[list_of_slices[i],:,:].T, cmap='hot',origin="lower", vmin=0, vmax=1, alpha = 1.0*(heatmap_data_GT[list_of_slices[i],:,:].T>0.4)) #Jo større tal ved alpha, jo bredere kurve  ax.set_title('dim, '+str(subject)+", Slice: "+str(i))
    
#     ax[i,2].imshow(img_data[list_of_slices[i],:,:].T,cmap="gray",origin="lower")
#     ax[i,2].imshow(heatmap_data_pred[list_of_slices[i],:,:].T, cmap='hot',origin="lower", vmin=0, vmax=1, alpha = 1.0*(heatmap_data_pred[list_of_slices[i],:,:].T>0.4)) #Jo større tal ved alpha, jo bredere kurve  ax.set_title('dim, '+str(subject)+", Slice: "+str(i))
    
# ax[0,0].set_title('Scan',fontsize=10)
# ax[0,1].set_title('GT',fontsize=10)
# ax[0,2].set_title('Prediction',fontsize=10)
# plt.show()
# plt.savefig()





### RH BOUNDING BOX ###





# Add a shared y-axis label
#fig.text(0.04, 0.5, 'Amplitude', va='center', rotation='vertical')

# Adjust spacing between subplots
#fig.subplots_adjust(wspace=0.00001)

# Show the figure

# show_heatmap_dim1(img_data, heatmap_data, no_slices=3000)
# show_heatmap_dim2(img_data, heatmap_data, no_slices=3000)
# show_heatmap_dim3(img_data, heatmap_data, no_slices=3000)

#dim1 = 31
#dim2 = 26 (eller 25)
#dim3 = 91