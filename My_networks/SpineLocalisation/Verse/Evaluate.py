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
from sklearn.metrics import mean_squared_error
import pandas as pd
#My own documents
from my_plotting_functions import *
from SpineLocalisationNet import *
from Create_dataset import LoadData
from my_data_utils import *
import importlib
import pickle

######################### CONTROL PANEL #########################
plot_loss = 0
plot_heatmaps = 1
#################################################################


#Plot loss
if plot_loss:
    ### ADJUST ###
    checkpoint_filename = 'First_try2950_batchsize1_lr0.0001_wd0.0005.pth' #'with_dataaugmentation_batchsize1_lr0.0001_wd0.0005.pth' #DET HER ER DEN GODE: 'checkpoint_batchsize1_lr0.0001_wd0.0005.pth'
    #checkpoint_filename = 'checkpoint_batchsize1_learningrate0.0001.pth'
    ##############

    #Define directories
    mac = '/Users/andreasaspe/Documents/Checkpoints/SpineLocalisation'
    GPUcluster = '/home/s174197/Checkpoints/SpineLocalisation'
    GPU1 = 'C:/Users/PC/Documents/Andreas_s174197/Thesis/My_code/My_networks/SpineLocalisation/Checkpoints'
    GPU2 = ''
    checkpoint_dir = os.path.join(mac,checkpoint_filename)
    
    #Call function
    Plot_loss(checkpoint_dir)

#Plot predictions
if plot_heatmaps:
    ### ADJUST ###
    data_type = 'test'
    heatmap_pred_dir = '/Users/andreasaspe/Documents/Data/Verse20/SpineLocalisation/Verse20_'+data_type+'_predictions' #'/Users/andreasaspe/Documents/Data/Verse20/Data_for_figures' #'/Users/andreasaspe/Documents/Data/Verse20/SpineLocalisation/Verse20_test_predictions' #Predictions or ground truth.
    heatmap_target_dir = '/Users/andreasaspe/Documents/Data/Verse20/SpineLocalisation/Verse20_'+data_type+'_heatmaps' #'/Users/andreasaspe/Documents/Data/Verse20/Data_for_figures' #'/Users/andreasaspe/Documents/Data/Verse20/SpineLocalisation/Verse20_test_predictions' #Predictions or ground truth.
    img_dir = '/Users/andreasaspe/Documents/Data/Verse20/SpineLocalisation/Verse20_'+data_type+'_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_training_unpacked' #'/Users/andreasaspe/Documents/Data/Verse20/SpineLocalisation/Verse20_test_prep/img' #Image folder of prepped data
    padding_specifications_dir = '/Users/andreasaspe/Documents/Data/Verse20/SpineLocalisation/Padding_specifications/pad_'+data_type #'/Users/andreasaspe/Documents/Data/Verse20/SpineLocalisation/Padding_specifications/pad_test'

    all_scans = 0 #Set to 1 if you want to preprocess all scans
    with open("/Users/andreasaspe/iCloud/DTU/12.semester/Thesis/MY_CODE/Other_scripts/list_of_subjects_"+data_type+"_VERSE", "rb") as fp:   # Unpickling
        list_of_subjects = pickle.load(fp)
    # list_of_subjects = ['sub-verse526','sub-verse809','sub-gl428'] #['sub-verse526'] #['sub-gl195'] #['sub-verse506','sub-verse525','sub-verse503'] #524 går ud over! sub-verse570 sub-verse764 'sub-verse578' sub-verse563 og 60 List of subjects, 'sub-verse510'512 #verse526 var før og efter dårlig med bounding box
    
    # list_of_subjects = ['sub-verse526','sub-verse809','sub-gl428'] #Highest MSE loss with worst first 
    # list_of_subjects = ['sub-verse712','sub-verse813','sub-verse801'] #Lowest MSE loss with best first 
    
    #Fejler boudning box!!
    # sub-gl195, fejler hvis area < 200 og man ikke normaliserer. Men måske derfor jeg skal køre COM som worst case.
    
    #Sketchy?
    #Måske lidt mere på z-aksen?
    #sub-verse809
    #sub-verse764 - too low heatmap values.
    #sub-verse710 ????
    # gl195!!!
    
    #Dårlig
    #gl428
    #gl279
    
    
    #God
    #verse517

    ##############

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

    subject_data = []
    mse_data = []
    iou_list = []
    
    for subject in all_subjects:
        print(subject)
        
        #Load predictions
        filename_heatmap = [f for f in listdir(heatmap_pred_dir) if f.startswith(subject)][0]
        heatmap_file_dir = os.path.join(heatmap_pred_dir, filename_heatmap)
        heatmap_data = torch.load(heatmap_file_dir, map_location=device)
        heatmap_data_prediction = heatmap_data.detach().numpy()
        #heatmap_data_prediction[heatmap_data_prediction > 1.3] = 0
        #Normalize
        heatmap_data_prediction_normalised = (heatmap_data_prediction - heatmap_data_prediction.min()) / (heatmap_data_prediction.max() - heatmap_data_prediction.min())

        #Load target
        filename_heatmap = [f for f in listdir(heatmap_target_dir) if f.startswith(subject)][0]
        heatmap_file_dir = os.path.join(heatmap_target_dir, filename_heatmap)
        heatmap_nib = nib.load(heatmap_file_dir)
        heatmap_data_target = np.asanyarray(heatmap_nib.dataobj, dtype=np.float32)
        # #Normalize
        # heatmap_data = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())
    

        
        filename_img = subject + "_img.nii.gz"
        img_nib = nib.load(os.path.join(img_dir,filename_img))
        img_data = np.asanyarray(img_nib.dataobj, dtype=np.float32)
        
        
        # Calculate the MSE loss
        mse = MSEloss(heatmap_data_target, heatmap_data_prediction)
        
        # Print the MSE loss
        # print("MSE loss:", mse)
        # print()
        
        subject_data.append(subject)
        mse_data.append(mse)

        # show_heatmap_img_dim1(img_data,heatmap_data_target,subject,alpha=0.1,no_slices=40)
        # show_slices_dim1(img_data,subject,no_slices=40)
        # show_heatmap_dim1(heatmap_data_prediction,subject,no_slices=40)
        # show_heatmap_dim1(heatmap_data_prediction_normalised,subject,no_slices=40)
        # show_heatmap_img_dim1(img_data,heatmap_data_prediction_normalised,subject,no_slices=40,alpha=0.3)
        # show_heatmap_img_dim1(img_data,heatmap_data_prediction,subject,no_slices=40,alpha=0.1)
        
        #The predictio

        # Find bounding box!
        with open(padding_specifications_dir, 'rb') as f:
            padding_specifications = pickle.load(f)
            
        restrictions = padding_specifications[subject]
                
        bb_coordinates_prediction, COM_prediction = BoundingBox(heatmap_data_prediction_normalised,restrictions)
        bb_coordinates_target, COM_target = BoundingBox(heatmap_data_target,restrictions)
        
        iou = calculate_iou(bb_coordinates_prediction,bb_coordinates_target)
        iou_list.append(iou)
        
        # if iou < 0.8:
        #     show_boundingboxes_dim1(img_data,bb_coordinates_prediction,bb_coordinates_target,subject)
        #     show_boundingboxes_dim2(img_data,bb_coordinates_prediction,bb_coordinates_target,subject)
        #     show_boundingboxes_dim3(img_data,bb_coordinates_prediction,bb_coordinates_target,subject)



        # try:
        #     bb_coordinates, COM = BoundingBox(heatmap_data_prediction,restrictions)
        # except:
            # bb_coordinates, COM = BoundingBoxFromCOM(heatmap_data_prediction.numpy(),restrictions)
        
        # show_COM_dim1(img_data, COM, subject, markersize=3)
        
        # print(COM)
        
        # show_slices_dim1(img_data,subject,no_slices=40)
        # show_boundingbox_dim1(img_data,bb_coordinates,subject)
        # show_boundingbox_dim1(img_data,bb_coordinates1,subject)
        # show_boundingbox_dim2(img_data,bb_coordinates,subject)
        # show_boundingbox_dim3(img_data,bb_coordinates,subject)
        #show_heatmap_dim1(img_data,heatmap_data,subject,alpha=0.4,no_slices=40)
        #show_heatmap_dim2(img_data,heatmap_data,subject,alpha=0.4,no_slices=40)
        #show_heatmap_dim3(img_data,heatmap_data,subject,alpha=0.4,no_slices=60)
        




df = pd.DataFrame({'subjects': subject_data, 'mse': mse_data})
df = df.sort_values(by='mse', ascending=True)

print("Average iou is {}".format(np.mean(iou_list)))
print("Minimum iou is {}".format(np.min(iou_list)))
print("Maximum iou is {}".format(np.max(iou_list)))



#Gammel load data
# filename_heatmap = [f for f in listdir(heatmap_dir) if f.startswith(subject)][0]
# heatmap_file_dir = os.path.join(heatmap_dir, filename_heatmap)
# try: #Load tensor or numpy
#     heatmap_data = torch.load(heatmap_file_dir, map_location=device)
#     heatmap_data = heatmap_data.detach().numpy()
#     #Normalize
#     heatmap_data = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())
# except:
#     heatmap_nib = nib.load(heatmap_file_dir)
#     heatmap_data = np.asanyarray(heatmap_nib.dataobj, dtype=np.float32)
#     #Normalize
#     heatmap_data = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())






#Fjernede også den her på et tidspunkt. Måske noget til COM?
        # heatmap_data_target = heatmap_data_target.numpy()
