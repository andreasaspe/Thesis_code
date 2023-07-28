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
import pandas as pd
#My own documents
from my_plotting_functions import *
from new_VertebraeLocalisationNet import *
from Create_dataset import LoadData
from my_data_utils import *
import importlib
import pickle

######################### CONTROL PANEL #########################
plot_loss = 0
predict_and_plot = 1
#################################################################


#Plot loss
if plot_loss:
    ### ADJUST ###
    checkpoint_filename = 'Train_and_validation_LeakyReluinfirstnetworkonly_normalinit1314.pth' #'FIXEDVAL_dropout3_higherdecay_batchsize1_lr1e-05_wd0.0001.pth'# 'New_preprocessing_batchsize1_lr6e-08_wd0.0001.pth' #'Adam_optimizerVALIDATION!_batchsize1_lr1e-08_wd0.0001.pth' #DET HER ER DEN GODE: 'checkpoint_batchsize1_lr0.0001_wd0.0005.pth'
    #checkpoint_filename = 'checkpoint_batchsize1_learningrate0.0001.pth'
    ##############

    #Define directories
    mac = '/Users/andreasaspe/Documents/Checkpoints/VertebraeLocalisation'
    GPUcluster = '/home/s174197/Checkpoints/VertebraeLocalisation'
    GPU1 = 'C:/Users/PC/Documents/Andreas_s174197/Thesis/My_code/My_networks/VertebraeLocalisation/Checkpoints'
    GPU2 = ''
    checkpoint_dir = os.path.join(mac,checkpoint_filename)
    
    #Call function
    Plot_loss(checkpoint_dir)

#Plot predictions
if predict_and_plot:
    ### ADJUST ###
    heatmap_pred_dir = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation/TO_REPORT/Verse20_test_predictions' #'/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation/ONLYONESAMPLE_Verse20_validation_predictions' #'/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation/Verse20_test_predictions' #Predictions or ground truth.
    heatmap_target_dir = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation/Verse20_test_heatmaps' #'/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation/Verse20_test_heatmaps' #Predictions or ground truth.
    img_dir = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation/Verse20_test_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation/Verse20_test_prep/img' #Image folder of prepped data


    all_scans = 0 #Set to 1 if you want to preprocess all scans
    list_of_subjects = ['sub-verse768'] #'sub-verse582','sub-verse810'] #524 går ud over! 'sub-gl279' sub-verse570 sub-verse764 'sub-verse578' sub-verse563 og 60 List of subjects, 'sub-verse510'512 #verse526 var før og efter dårlig med bounding box
    ##############

    
    #sub-gl090 - har den alle centroids??? Det har den sgu da ikke...
    #SUB-VERSE824 VAR FORKERT FØR.. SE IGEN


    #Define list of scans
    if all_scans:
        all_subjects = []
        for filename in listdir(img_dir):
            subject = filename.split("_")[0]
            if subject.find('verse') != -1: #PLOTTER KUN VERSE. IKKE GL
                all_subjects.append(subject)
        all_subjects = np.unique(all_subjects)
        all_subjects = all_subjects[all_subjects != '.DS']
    else:
        all_subjects = list_of_subjects

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #device = torch.device('cpu')

    mse_data = []
    subject_data = []

    for subject in tqdm(all_subjects):
        print(subject)
        print("\n\n\n")

        #Load predictions
        filename_heatmap = [f for f in listdir(heatmap_pred_dir) if f.startswith(subject)][0]
        heatmap_file_dir = os.path.join(heatmap_pred_dir, filename_heatmap)
        heatmap_data = torch.load(heatmap_file_dir, map_location=device)
        heatmap_data_prediction = heatmap_data.detach().squeeze().numpy()
        # #Normalize
        # heatmap_data_prediction = (heatmap_data_prediction - heatmap_data_prediction.min()) / (heatmap_data_prediction.max() - heatmap_data_prediction.min())

        #Load target
        filename_heatmap = [f for f in listdir(heatmap_target_dir) if f.startswith(subject)][0]
        heatmap_file_dir = os.path.join(heatmap_target_dir, filename_heatmap)
        heatmap_nib = nib.load(heatmap_file_dir)
        heatmap_data_target = np.asanyarray(heatmap_nib.dataobj, dtype=np.float32)
        # #Normalize
        # heatmap_data = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())

        mse = MSEloss(heatmap_data_target,heatmap_data_prediction)
        
        #Append to list
        subject_data.append(subject)
        mse_data.append(mse)
        
        
        #Load image
        filename_img = subject + "_img.nii.gz"
        img_nib = nib.load(os.path.join(img_dir,filename_img))
        img_data = np.asanyarray(img_nib.dataobj, dtype=npz.float32)
        
        for i in range(8):
            show_heatmap_img_dim1(img_data,heatmap_data_target[i,:,:,:],subject,no_slices=15,alpha=0.1)
            show_heatmap_img_dim1(img_data,heatmap_data_prediction[i,:,:,:],subject,no_slices=15,alpha=0.1)
            # show_heatmap_dim1(heatmap_data_prediction[i,:,:,:],subject)


        # print("Done")

df_stage2 = pd.DataFrame({'subjects': subject_data, 'mse': mse_data})
df_stage2 = df_stage2.sort_values(by='mse', ascending=True)

#Get median case
median_idx = int(np.round(len(df_stage2)/2))-1 #Fordi det er 0 indexeret!
#Extract median case
median_subject = df_stage2.iloc[median_idx].subjects
print("The median subject is {}".format(median_subject))
            
# df_stage1.to_csv(os.path.join(predictions_folder,'df_stage1.csv'), index=False)






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

