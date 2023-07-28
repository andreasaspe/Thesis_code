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
from new_VertebraeLocalisationNet import *
from Create_dataset import LoadData
from my_data_utils import *
import importlib
import pickle

######################### CONTROL PANEL #########################
plot_loss = 1
predict_and_plot = 0
#################################################################


#Plot loss
if plot_loss:
    ### ADJUST ###
    checkpoint_filename = 'Only_rotation_earlystopping_epoch1790_batchsize1_lr1e-05_wd0.0001.pth' #'FIXEDVAL_dropout3_higherdecay_batchsize1_lr1e-05_wd0.0001.pth'# 'New_preprocessing_batchsize1_lr6e-08_wd0.0001.pth' #'Adam_optimizerVALIDATION!_batchsize1_lr1e-08_wd0.0001.pth' #DET HER ER DEN GODE: 'checkpoint_batchsize1_lr0.0001_wd0.0005.pth'
    #checkpoint_filename = 'checkpoint_batchsize1_learningrate0.0001.pth'
    ##############

    #Define directories
    mac = '/Users/andreasaspe/Documents/Checkpoints/VertebraeLocalisation2'
    GPUcluster = '/home/s174197/Checkpoints/VertebraeLocalisation2'
    GPU1 = 'C:/Users/PC/Documents/Andreas_s174197/Thesis/My_code/My_networks/VertebraeLocalisation2/Checkpoints'
    GPU2 = ''
    checkpoint_dir = os.path.join(mac,checkpoint_filename)
    
    #Call function
    Plot_loss(checkpoint_dir)

#Plot predictions
if predict_and_plot:
    ### ADJUST ###
    # heatmap_pred_dir = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation2/Verse20_test_predictions' #'/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation2/Verse20_validation_predictions_justforfun' #'/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation2/Verse20_validation_predictions2' #'/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation2/SUBSET_Verse20_validation_predictions_alldata'  #'/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation2/Working_code/Verse20_validation_predictions2' #Predictions or ground truth.
    heatmap_target_dir = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation2/Verse20_test_heatmaps' #'/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation2/Verse20_validation_heatmaps' #'/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation2/Working_code/Verse20_validation_heatmaps2''/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation2/SUBSET_Verse20_validation_heatmaps_alldata' #'/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation2/Working_code/Verse20_validation_heatmaps2' #Predictions or ground truth.
    img_dir = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation2/Verse20_test_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation2/Verse20_validation_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation2/SUBSET_Verse20_validation_prep_alldata/img' #'/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation2/Working_code/Verse20_validation_prep2/img' #Image folder of prepped data

    #Old good
    # heatmap_pred_dir = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation2/Verse20_validation_predictions_justforfun'
    # heatmap_target_dir = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation2/Verse20_validation_heatmaps2'
    # img_dir = '/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation2/Verse20_validation_prep2/img'


    all_scans = 1 #Set to 1 if you want to preprocess all scans
    list_of_subjects = ['sub-verse511'] #verse505 gl279 524 går ud over! sub-verse570 sub-verse764 'sub-verse578' sub-verse563 og 60 List of subjects, 'sub-verse510'512 #verse526 var før og efter dårlig med bounding box
    ##############

    
    #sub-gl090 - har den alle centroids??? Det har den sgu da ikke...
    #SUB-VERSE824 VAR FORKERT FØR.. SE IGEN


    #Define list of scans
    if all_scans:
        all_subjects = []
        for filename in listdir(heatmap_target_dir): #img_dir
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

        # #Load predictions
        # filename_heatmap = [f for f in listdir(heatmap_pred_dir) if f.startswith(subject)][0]
        # heatmap_file_dir = os.path.join(heatmap_pred_dir, filename_heatmap)
        # heatmap_data = torch.load(heatmap_file_dir, map_location=device)
        # heatmap_data_prediction = heatmap_data.detach().numpy()
        # #heatmap_data_prediction[heatmap_data_prediction > 1.3] = 0
        # #Normalize
        # heatmap_data_prediction = (heatmap_data_prediction - heatmap_data_prediction.min()) / (heatmap_data_prediction.max() - heatmap_data_prediction.min())

        #Load target
        filename_heatmap = [f for f in listdir(heatmap_target_dir) if f.startswith(subject)][0]
        heatmap_file_dir = os.path.join(heatmap_target_dir, filename_heatmap)
        heatmap_nib = nib.load(heatmap_file_dir)
        heatmap_data_target = np.asanyarray(heatmap_nib.dataobj, dtype=np.float32)
        # #Normalize
        # heatmap_data = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())

        
        #Load image
        filename_img = subject + "_img.nii.gz"
        img_nib = nib.load(os.path.join(img_dir,filename_img))
        img_data = np.asanyarray(img_nib.dataobj, dtype=np.float32)
        
        print("done")

        show_heatmap_img_dim1(img_data,heatmap_data_target[:,:,:],subject,no_slices=100,alpha=0.3)
        show_heatmap_img_dim2(img_data,heatmap_data_target[:,:,:],subject,no_slices=100,alpha=0.3)

        
        continue
        # show_heatmap_img_dim1(img_data,heatmap_data_prediction[0,0,:,:,:],subject,no_slices=20,alpha=0.2)
        # show_heatmap_dim1(heatmap_data_prediction[0,0,:,:,:],subject,no_slices=100,alpha=0.3)
        #show_slices_dim1(img_data,subject)
        









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

