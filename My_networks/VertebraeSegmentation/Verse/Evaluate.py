#General imports
import os
import sys
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
import nibabel as nib
#My own documents
from my_plotting_functions import *
from VertebraeSegmentationNet import *
from Create_dataset import LoadData
from my_data_utils import *
import importlib
import pickle
import cv2

######################### CONTROL PANEL #########################
plot_loss = 0
predict_and_plot = 1
#################################################################


#Plot loss
if plot_loss:
    ### ADJUST ###
    checkpoint_filename = 'FIXEDVAL_dropout3_higherdecay_batchsize1_lr1e-05_wd0.0001.pth'# 'New_preprocessing_batchsize1_lr6e-08_wd0.0001.pth' #'Adam_optimizerVALIDATION!_batchsize1_lr1e-08_wd0.0001.pth' #DET HER ER DEN GODE: 'checkpoint_batchsize1_lr0.0001_wd0.0005.pth'
    #checkpoint_filename = 'checkpoint_batchsize1_learningrate0.0001.pth'
    ##############

    #Define directories
    mac = '/Users/andreasaspe/Documents/Checkpoints/VertebraeSegmentation'
    GPUcluster = '/home/s174197/Checkpoints/VertebraeSegmentation'
    GPU1 = 'C:/Users/PC/Documents/Andreas_s174197/Thesis/My_code/My_networks/VertebraeSegmentation/Checkpoints'
    GPU2 = ''
    checkpoint_dir = os.path.join(mac,checkpoint_filename)
    
    #Call function
    Plot_loss(checkpoint_dir)

#Plot predictions
if predict_and_plot:
    ### ADJUST ###
    #Outputs
    msk_pred_dir = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeSegmentation/Verse20_validation_predictions' #Predictions or ground truth.
    msk_target_dir = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeSegmentation/Verse20_validation_prep/msk' #Predictions or ground truth.
    #Inputs
    img_dir = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeSegmentation/Verse20_validation_prep/img' #Image folder of prepped data
    heatmap_dir = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeSegmentation/Verse20_validation_prep/heatmaps' #Image folder of prepped data


    all_scans = 1 #Set to 1 if you want to preprocess all scans
    list_of_subjects = ['sub-verse505-19'] #524 går ud over! sub-verse570 sub-verse764 'sub-verse578' sub-verse563 og 60 List of subjects, 'sub-verse510'512 #verse526 var før og efter dårlig med bounding box
    ##############
    
    
    #sub-gl090 - har den alle centroids??? Det har den sgu da ikke...
    #SUB-VERSE824 VAR FORKERT FØR.. SE IGEN


    #Define list of scans
    if all_scans:
        all_subjects = []
        for filename in listdir(img_dir):
            subject = filename.split("_")[0]
            #if subject.find('verse') != -1: #PLOTTER KUN VERSE. IKKE GL
            all_subjects.append(subject)
        all_subjects = np.unique(all_subjects)
        all_subjects = all_subjects[all_subjects != '.DS']
    else:
        all_subjects = list_of_subjects

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #device = torch.device('cpu')


    for subject in tqdm(all_subjects):
        print(subject)
        print("\n\n\n")
        
        #Load prediction mask
        # filename_msk_pred = [f for f in listdir(msk_pred_dir) if f.startswith(subject)][0]
        # file_dir_msk_pred = os.path.join(msk_pred_dir, filename_msk_pred)
        # msk_pred = torch.load(file_dir_msk_pred, map_location=device)

        #Load target mask
        filename_msk_target = [f for f in listdir(msk_target_dir) if f.startswith(subject)][0]
        file_dir_msk_target = os.path.join(msk_target_dir, filename_msk_target)
        msk_target = np.load(file_dir_msk_target)

        #Load input image
        filename_img = [f for f in listdir(img_dir) if f.startswith(subject)][0]
        file_dir_img = os.path.join(img_dir, filename_img)
        img_data = np.load(file_dir_img)
        
        #Load input heatmap
        filename_heatmap = [f for f in listdir(heatmap_dir) if f.startswith(subject)][0]
        file_dir_heatmap = os.path.join(heatmap_dir, filename_heatmap)
        heatmap_data = np.load(file_dir_heatmap)
                
        #show_mask_img_dim1(img_data,msk_target,subject,no_slices=1000)
        # show_mask_dim1(msk_target,subject,no_slices=1000)
        # show_mask_dim2(msk_target,subject,no_slices=1000)
        # show_mask_dim3(msk_target,subject,no_slices=1000)
        
        binary_mask = np.uint8(np.flip(msk_target[:,:,51].T))
        
        # Find contours
        contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_image = np.zeros((128,128))
        # Draw contours on the blank image
        #cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)  # Green color, thickness 2
        
        # Draw contours on the blank image
        cv2.drawContours(contour_image, contours, -1, 255, 2)  # White color, thickness 2
        
        # Display the image with contours using Matplotlib
        plt.imshow(contour_image, cmap='gray')
        plt.axis('off')
        plt.show()
        break
        
        
# # Filter contours based on desired criteria
# filtered_contours = []
# for contour in contours:
#     # Calculate contour properties (e.g., area, bounding box)
#     area = cv2.contourArea(contour)
#     x, y, width, height = cv2.boundingRect(contour)
#     aspect_ratio = width / height

#     # Apply filtering criteria (example: retain contours with certain area and aspect ratio)
#     if area > 1000 and aspect_ratio > 1.5:
#         filtered_contours.append(contour)

# # Create a new mask with only the selected contours
# new_mask = np.zeros_like(binary_mask)
# cv2.drawContours(new_mask, filtered_contours, -1, (255), thickness=cv2.FILLED)

#         # show_mask_img_dim1(img_data,msk_pred[0,0,:,:,:],subject)










#     #Load data
#     dataset = LoadData(img_dir=img_dir, heatmap_dir=heatmap_dir)
#     dataloader = DataLoader(dataset, batch_size=1,
#                             shuffle=True, num_workers=0) #SET TO True!

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #device = torch.device('cpu')
#     checkpoint = torch.load(checkpoint_dir,map_location=device)
    
#     #Define model
#     model = VertebraeLocalisationNet(0.0)
#     # Load the saved weights into the model
#     model.load_state_dict(checkpoint['model_state_dict'])
    
#     # Set the model to evaluation mode
#     model.eval() 
    
#     loss_fn = nn.MSELoss()
    
#     with torch.no_grad():
#         for inputs, targets, subject in tqdm(dataloader):
#             assert len(subject) == 1 #Make sure we are only predicting one batch
#             print()
#             print(subject[0])
#             predictions = model(inputs)
#             loss = loss_fn(predictions, targets)
#             predictions = predictions.squeeze()
#             predictions = predictions.detach().numpy()
            
#             targets = targets.squeeze()
#             inputs = inputs.squeeze(0,1)
            
#             # for i in range(8):
#             #     predictions[i,:,:,:] = (predictions[i,:,:,:] - predictions[i,:,:,:].min()) / (predictions[i,:,:,:].max() - predictions[i,:,:,:].min())
      
#             for i in range(8): #3,5
#                 if torch.max(targets[i,:,:,:])==1:
#                     #show_heatmap_img_dim1(inputs,targets[i,:,:,:],subject[0],no_slices=40)
#                     # show_heatmap_dim1(predictions[i,:,:,:],subject[0],alpha=0.0001,no_slices=40)
#                     show_heatmap_img_dim1(inputs,predictions[i,:,:,:],subject[0],no_slices=40)

             
#             print()
#             print(loss.item())
#             print()
            
#             # show_heatmap_dim1(targets[0,:,:,:],subject,no_slices=40)
#             # show_heatmap_dim1(predictions[0,:,:,:],subject,no_slices=40)

            
        
#     show_slices_dim1(img_data,subject,no_slices=40)

#         #show_heatmap_dim1(img_data,heatmap_data,subject,alpha=0.4,no_slices=40)
#         #show_heatmap_dim2(img_data,heatmap_data,subject,alpha=0.4,no_slices=40)
#         #show_heatmap_dim3(img_data,heatmap_data,subject,alpha=0.4,no_slices=60)

















# #from My_networks.VertebraeLocalisation.Verse.VertebraeLocalisationNet import *

