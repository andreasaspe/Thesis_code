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
import importlib
import pickle
from tqdm import tqdm
import pandas as pd
#My own documents
from my_plotting_functions import *
from Load_data import *
from my_data_utils import *
#Import networks
from SpineLocalisationNet import SpineLocalisationNet
# from new_VertebraeLocalisationNet import VertebraeLocalisationNet
from new_VertebraeLocalisationNet_batchnormdropout import VertebraeLocalisationNet
from VertebraeSegmentationNet import VertebraeSegmentationNet


#######################################################
#################### CONTROL PANEL ####################
#######################################################
#Define scans
all_scans = 0 #SET TO ZERO!! We want to process list_of_subject!!!!!!!!
list_of_subjects = ['VERTEBRAE_LOWHU_0147_SERIES0013'] #['VERTEBRAE_FRACTURE_0239_SERIES0003','VERTEBRAE_LOWHU_0101_SERIES0012'] #['VERTEBRAE_HEALTHY_0001_SERIES0010'] #List of subjects
with open("E:\s174197\Thesis\My_code\Other_scripts\list_of_subjects_LOWHU", "rb") as fp:   # Unpickling
    list_of_subjects = pickle.load(fp)

# VERTEBRAE_LOWHU_0100_SERIES0018.

#Define directories
dir_data_original = r'G:\DTU-Vertebra-1\NIFTI'
dir_data_stage1 = r'E:\s174197\data_RH\SpineLocalisation\data_prep\img' #'/zhome/bb/f/127616/Documents/Thesis/Rawdata_training'
dir_data_stage2 = r'E:\s174197\data_RH\VertebraeLocalisation2\data_prep\img' #'/zhome/bb/f/127616/Documents/Thesis/Rawdata_training'
dir_data_stage3 = r'E:\s174197\data_RH\VertebraeSegmentation\data_prep\img' #'/zhome/bb/f/127616/Documents/Thesis/Rawdata_training'
checkpoint_dir_stage1 = r'E:\s174197\Checkpoints\SpineLocalisation\First_try2650_batchsize1_lr0.0001_wd0.0005.pth' #r'E:\Andreas_s174197\Thesis\My_code\My_networks\Spine_Localisation\Checkpoints\checkpoint_batchsize1_lr0.0001_wd0.0005.pth'
checkpoint_dir_stage2 = r'E:\s174197\Checkpoints\VertebraeLocalisation2\Only_elastic_earlystopping_epoch1400_batchsize1_lr1e-05_wd0.0001.pth' #First_try_epoch430_batchsize1_lr1e-05_wd0.0001.pth' #r'E:\s174197\Checkpoints\VertebraeLocalisation2\First_try_epoch430_batchsize1_lr1e-05_wd0.0001.pth' #Batchnorm_dropout_batchsize1_lr1e-05_wd0.0001.pth' #First_try_epoch430_batchsize1_lr1e-05_wd0.0001.pth'#r'E:\s174197\Checkpoints\VertebraeLocalisation2\no_tanh_no_init_batchsize1_lr1e-05_wd0.0001.pth' #r'E:\s174197\Checkpoints\VertebraeLocalisation2\First_try_epoch430_batchsize1_lr1e-05_wd0.0001.pth' #r'E:\Andreas_s174197\Thesis\My_code\My_networks\Spine_Localisation\Checkpoints\checkpoint_batchsize1_lr0.0001_wd0.0005.pth'
checkpoint_dir_stage3 = r'E:\s174197\Checkpoints\VertebraeSegmentation\only_elastic2_again_step33650_batchsize1_lr1e-05_wd0.0005.pth' #r'E:\s174197\Checkpoints\VertebraeSegmentation\First_try_step8450_batchsize1_lr1e-05_wd0.0005.pth'
dir_predictions = 'E:/s174197/data_RH/Predictions'
# save_dataframe_to_folder = r'E:\s174197\Thesis\MY_CODE\Other_scripts'

#Define rescale and reorientation parameters
New_orientation = ('L', 'A', 'S')
New_voxel_size = 8 # [mm]

#For stage 2
dim1_new = 96
dim2_new = 96
dim3_new = 128
pad_value = -1
#######################################################
#######################################################
#######################################################

def calculate_mean_within_sphere(volume, center_x, center_y, center_z, diameter):
    # Calculate the radius of the sphere from the given diameter
    radius = diameter // 2

    # Generate coordinates grid for the volume
    x, y, z = np.indices(volume.shape)

    # Calculate the Euclidean distance from each voxel to the sphere center
    distances = np.sqrt((x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2)

    # Create a binary mask indicating the voxels within the sphere
    mask = distances <= radius

    # Calculate the mean value of the voxels within the sphere
    mean_value = np.mean(volume[mask])

    return mean_value, mask

# def calculate_mean_within_spuare(volume, center_x, center_y, center_z, diameter):
#     # Calculate the radius of the sphere from the given diameter
#     radius = diameter // 2

#     # Generate coordinates grid for the volume
#     x, y, z = np.indices(volume.shape)

#     # Calculate the Euclidean distance from each voxel to the sphere center
#     volume = 
#     distances = np.sqrt((x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2)

#     # Create a binary mask indicating the voxels within the sphere
#     mask = distances <= radius

#     # Calculate the mean value of the voxels within the sphere
#     mean_value = np.mean(volume[mask])

#     return mean_value, mask


#Load file with metadata as pd.dataframe
df = pd.read_excel("E:\s174197\Thesis\MY_CODE\Other_scripts\HU_value_list.xlsx")

subject_list = []
l1_hu_predict_list = []
l1_hu_target_list = []

#Define list of scans
if all_scans:
    all_subjects = []
    for filename in listdir(dir_data_stage1):
        subject = filename.split("-")[0]
        # if subject.find('VERTEBRAE_LOWHU_0100') != -1: #PLOTTER KUN VERSE. IKKE GL
        all_subjects.append(subject)
    all_subjects = np.unique(all_subjects)
    #Sorterer fil '.DS' fra
    all_subjects = all_subjects[all_subjects != '.DS']
else:
    all_subjects = list_of_subjects
    
# goon = False

for subject in tqdm(all_subjects):
    
    # print('"'+subject+'": ')
    # # continue

    # if subject == 'VERTEBRAE_FRACTURE_0327_SERIES0011':
    #     goon = True
    
    # if goon == False:
    #     continue
    
    # print(subject+",")
    # continue
    #Load original image
    img_nib_original = nib.load(os.path.join(dir_data_original,subject+'.nii.gz'))
    data_original = np.asanyarray(img_nib_original.dataobj, dtype=np.float32) #Maybe remove?
    # show_slices_dim1(data_original,subject,no_slices=60)
    
    # continue

    #Define zooms
    original_zooms = img_nib_original.header.get_zooms()
    
    #Stage 1
    filename_img = subject + "-img.nii.gz"
    img_nib = nib.load(os.path.join(dir_data_stage1,filename_img))
    data_img = np.asanyarray(img_nib.dataobj, dtype=np.float32)
    
    #Do padding
    data_img, restrictions = center_and_pad(data=data_img, new_dim=(64,64,128), pad_value=-1)
    
    # show_slices_dim1(data_img,subject)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #device = torch.device('cpu')
    checkpoint = torch.load(checkpoint_dir_stage1,map_location=device)

    #Define model
    model = SpineLocalisationNet()
    #Send to GPU
    model.to(device)
    # Load the saved weights into the model
    model.load_state_dict(checkpoint['model_state_dict'])
    
    data_img = torch.from_numpy(data_img)
    data_img = data_img.unsqueeze(0) 
    data_img = data_img.unsqueeze(0)
    
    #send to device
    data_img = data_img.to(device)
        
    # Set the model to evaluation mode
    model.eval() 
    with torch.no_grad():
        predictions = model(data_img)
        
    #Change formatting
    predictions = predictions.squeeze()
    predictions = predictions.cpu().detach().numpy()
    
    # show_heatmap_img_dim1(data_img.squeeze().cpu().detach().numpy(),predictions,subject)

    #Normalise
    predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min())

    # show_heatmap_img_dim1(data_img.squeeze().cpu().detach().numpy(),predictions,subject)

    bb_coordinates, COM = BoundingBox(predictions,restrictions)
    
    #Normal format
    data_img = data_img.squeeze(0,1).cpu().detach().numpy()

    #show_slices_dim1(data_img,no_slices=40)
    # show_boundingbox_dim1(data_img,bb_coordinates,subject)
    
    #Define zooms
    old_zooms = (8,8,8)
    new_zooms = original_zooms
    #Rescale bounding box
    original_bb_coordinates, original_COM = RescaleBoundingBox(new_zooms,old_zooms,bb_coordinates,COM,restrictions,borders=data_original.shape)
    
    # show_boundingbox_dim1(img_nib_original.get_fdata(),original_bb_coordinates,subject)

    #Rescale again
    new_zooms = (2,2,2)
    old_zooms = original_zooms
    new_bb_coordinates, new_COM = RescaleBoundingBox(new_zooms,old_zooms,original_bb_coordinates,original_COM)

    #Load data for stage 2
    filename_img = subject + "-img.nii.gz"
    img_nib = nib.load(os.path.join(dir_data_stage2,filename_img))
    data_img = np.asanyarray(img_nib.dataobj, dtype=np.float32)
    
    #Crop from BOUNDING BOX
    x_min, x_max, y_min, y_max, z_min, z_max = new_bb_coordinates
    x_min = np.round(x_min).astype(int)
    x_max = np.round(x_max).astype(int)
    y_min = np.round(y_min).astype(int)
    y_max = np.round(y_max).astype(int)
    z_min = np.round(z_min).astype(int)
    z_max = np.round(z_max).astype(int)
    x_range = [x_min,x_max]
    y_range = [y_min,y_max]
    z_range = [z_min,z_max]
    data_img = data_img[x_range[0]:x_range[1]+1,y_range[0]:y_range[1]+1,z_range[0]:z_range[1]+1]
    
    #Do padding
    data_img, restrictions = center_and_pad(data=data_img, new_dim=(96,96,128), pad_value=-1)
    #Update restrictions for above cropping
    restrictions = tuple(restrictions - np.array([x_min,x_max,y_min,y_max,z_min,z_max]))
    
    #show_slices_dim1(data_img,subject)
    
    
    
    #Predictions stage 2
    #Load checkpoint
    checkpoint = torch.load(checkpoint_dir_stage2,map_location=device)

    #Define model
    model = VertebraeLocalisationNet(0.0)
    
    #Do predictions (everything else regarding sending to GPU and such is handled inside the function)
    predictions = prepare_and_predict_VLN2(data_img, model, checkpoint)

    # predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min())
    # show_heatmap_img_dim1(data_img,predictions,subject,alpha=0.1,no_slices=20)

    
    ctd_list = find_centroids2(predictions)
    
    # show_centroids_new_dim1(data_img,ctd_list,subject,markersize=2,no_slices=20)
    
    # continue

    #Get mapping
    list_of_visible_vertebrae = mapping_RH[subject]
    
    #Convert to verse format from mapping
    ctd_list = centroids_to_verse(ctd_list,list_of_visible_vertebrae)

    # show_centroids_dim1(data_img,ctd_list,subject,text=1,markersize=2,no_slices=2)
    # show_centroids_RH_toreport_dim1(data_img,ctd_list,subject,text=1,markersize=2,no_slices=40)
    # continue
    
        
    
    # Lets rescale it back!
    new_zooms = original_zooms
    old_zooms = (2,2,2)
    ctd_list = RescaleCentroids_verse(new_zooms, old_zooms, ctd_list, restrictions)
    #Lets plot!
    # show_centroids_new_dim1(data_original,10,ctd_list,subject)

        
    for i, centroid in enumerate(ctd_list[1:]):

        v_number = int(centroid[0])
        v_name = number_to_name[v_number]
        
        if v_name == 'L1':
            # Set the center coordinates of the sphere
            x, y, z = centroid[1:]
            x = np.round(x).astype(int)
            y = np.round(y).astype(int)
            z = np.round(z+5/original_zooms[2]).astype(int)
            
            gap = 3
            
            offset_x = int(np.round(gap/original_zooms[0]))
            offset_y = int(np.round(gap/original_zooms[1]))
            offset_z = int(np.round(gap/original_zooms[2]))
    
            coordinates = (x-offset_x, x+offset_x, y-offset_y, y+offset_y, z-offset_z, z+offset_z)
    
            # print(subject)
            
            show_boundingbox_dim1(data_original, coordinates,subject,zooms=original_zooms,convert=1,linewidth=1,no_slices=2)
            # show_boundingbox_dim2(data_original, coordinates,subject,zooms=original_zooms,convert=0,linewidth=1)
            # show_boundingbox_dim3(data_original, coordinates,subject,zooms=original_zooms,convert=0,linewidth=1)
        
            l1_hu_predict = Calculate_HU_from_BoundingBox(data_original, coordinates) #Median value
            
            #Get HU value from meta_datafile
            # temp = subject.split("_") #Split into substrings
            # subject_without_ending = "_".join(temp[:-1]) #Subject name without SERIES. So we have VERTEBRAE_LOWHU_0100_SERIES0018 -> VERTEBRAE_LOWHU_0100
            
            filtered_df = df[df['Subject'] == subject]
            l1_hu_target = filtered_df['Median'].values[0]
            
            #Append lists
            subject_list.append(subject)
            l1_hu_target_list.append(l1_hu_target)
            l1_hu_predict_list.append(l1_hu_predict)
            
            print(subject,l1_hu_target,l1_hu_predict)

# # Create the DataFrame using the lists
# df_HU = pd.DataFrame({'Subject': subject_list,
#             'Target L1 HU': l1_hu_target_list,
#             'Prediction L1 HU': l1_hu_predict_list})

# df_HU.to_csv(os.path.join(dir_predictions,'HU_pred.csv'), index=False)

# df_HU.to_excel('HU_output.xlsx', index=False)

        # Set the diameter of the sphere
        # diameter = 15

        # # Call the function to calculate the mean value within the sphere
        # mean_value, mask = calculate_mean_within_sphere(data_original, center_x, center_y, center_z, diameter)
        
        # msk_data = mask*1
        
        # show_mask_img_dim1(data_original[210:260,:,:], msk_data[210:260,:,:], subject, no_slices=40)
    
































































































# #Load file with metadata as pd.dataframe
# df = pd.read_excel("G:\DTU-Vertebra-1\Metadata\DTU-vertebra-1-clinical-data.xlsx")

# subject_list = []
# l1_hu_predict_list = []
# l1_hu_target_list = []

# #Define list of scans
# if all_scans:
#     all_subjects = []
#     for filename in listdir(dir_data_stage1):
#         subject = filename.split("-")[0]
#         if subject.find('VERTEBRAE_LOWHU_0100') != -1: #PLOTTER KUN VERSE. IKKE GL
#             all_subjects.append(subject)
#     all_subjects = np.unique(all_subjects)
#     #Sorterer fil '.DS' fra
#     all_subjects = all_subjects[all_subjects != '.DS']
# else:
#     all_subjects = list_of_subjects
    

# for subject in tqdm(all_subjects):
    
#     # print(subject+",")
#     # continue
#     #Load original image
#     img_nib_original = nib.load(os.path.join(dir_data_original,subject+'.nii.gz'))
#     data_original = np.asanyarray(img_nib_original.dataobj, dtype=np.float32) #Maybe remove?
#     # show_slices_dim1(data_original,subject,no_slices=60)
    
#     # continue

#     #Define zooms
#     original_zooms = img_nib_original.header.get_zooms()
    
#     #Stage 1
#     filename_img = subject + "-img.nii.gz"
#     img_nib = nib.load(os.path.join(dir_data_stage1,filename_img))
#     data_img = np.asanyarray(img_nib.dataobj, dtype=np.float32)
    
#     #Do padding
#     data_img, restrictions = center_and_pad(data=data_img, new_dim=(64,64,128), pad_value=-1)
    
#     # show_slices_dim1(data_img,subject)
    
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #device = torch.device('cpu')
#     checkpoint = torch.load(checkpoint_dir_stage1,map_location=device)

#     #Define model
#     model = SpineLocalisationNet()
#     #Send to GPU
#     model.to(device)
#     # Load the saved weights into the model
#     model.load_state_dict(checkpoint['model_state_dict'])
    
#     data_img = torch.from_numpy(data_img)
#     data_img = data_img.unsqueeze(0) 
#     data_img = data_img.unsqueeze(0)
    
#     #send to device
#     data_img = data_img.to(device)
        
#     # Set the model to evaluation mode
#     model.eval() 
#     with torch.no_grad():
#         predictions = model(data_img)
        
#     #Change formatting
#     predictions = predictions.squeeze()
#     predictions = predictions.cpu().detach().numpy()
    
#     #show_heatmap_dim1(predictions,subject)
    
#     bb_coordinates, COM = BoundingBox(predictions,restrictions)
    
#     #Normal format
#     data_img = data_img.squeeze(0,1).cpu().detach().numpy()

#     #show_slices_dim1(data_img,no_slices=40)
#     # show_boundingbox_dim1(data_img,bb_coordinates,subject)
    
#     #Define zooms
#     old_zooms = (8,8,8)
#     new_zooms = original_zooms
#     #Rescale bounding box
#     original_bb_coordinates, original_COM = RescaleBoundingBox(new_zooms,old_zooms,bb_coordinates,COM,restrictions,borders=data_original.shape)
    
#     # show_boundingbox_dim1(img_nib_original.get_fdata(),original_bb_coordinates,subject)

#     #Rescale again
#     new_zooms = (2,2,2)
#     old_zooms = original_zooms
#     new_bb_coordinates, new_COM = RescaleBoundingBox(new_zooms,old_zooms,original_bb_coordinates,original_COM)

#     #Load data for stage 2
#     filename_img = subject + "-img.nii.gz"
#     img_nib = nib.load(os.path.join(dir_data_stage2,filename_img))
#     data_img = np.asanyarray(img_nib.dataobj, dtype=np.float32)
    
#     #Crop from BOUNDING BOX
#     x_min, x_max, y_min, y_max, z_min, z_max = new_bb_coordinates
#     x_min = np.round(x_min).astype(int)
#     x_max = np.round(x_max).astype(int)
#     y_min = np.round(y_min).astype(int)
#     y_max = np.round(y_max).astype(int)
#     z_min = np.round(z_min).astype(int)
#     z_max = np.round(z_max).astype(int)
#     x_range = [x_min,x_max]
#     y_range = [y_min,y_max]
#     z_range = [z_min,z_max]
#     data_img = data_img[x_range[0]:x_range[1]+1,y_range[0]:y_range[1]+1,z_range[0]:z_range[1]+1]
    
#     #Do padding
#     data_img, restrictions = center_and_pad(data=data_img, new_dim=(96,96,128), pad_value=-1)
#     #Update restrictions for above cropping
#     restrictions = tuple(restrictions - np.array([x_min,x_max,y_min,y_max,z_min,z_max]))
    
#     #show_slices_dim1(data_img,subject)
    
    
    
#     #Predictions stage 2
#     #Load checkpoint
#     checkpoint = torch.load(checkpoint_dir_stage2,map_location=device)

#     #Define model
#     model = VertebraeLocalisationNet(0.0)
    
#     #Do predictions (everything else regarding sending to GPU and such is handled inside the function)
#     predictions = prepare_and_predict_VLN2(data_img, model, checkpoint)

#     # show_heatmap_img_dim1(data_img,predictions,subject,alpha=0.2)

    
#     ctd_list = find_centroids(predictions)
    
#     # show_centroids_new_dim1(data_img,ctd_list,subject,markersize=2,no_slices=1000)
    
#     # #Lets define bounding box
#     # gap = 2 #2x2x2 mm
#     # bounding_boxes = []
#     # for i, centroid in enumerate(ctd_list): 
#     #     x, y, z= centroid
#     #     old_coordinates = (x-gap, x+gap, y-gap, y+gap, z-gap, z+gap)
        
#     #     # show_boundingbox_dim1(data_img, old_coordinates,subject,linewidth=1)
        
#     #     new_zooms = original_zooms
#     #     old_zooms = (2,2,2)
        
#     #     bb_coordinates = RescaleBoundingBox(new_zooms,old_zooms,old_coordinates)
        
#     #     show_boundingbox_dim1(data_original, bb_coordinates,subject,zooms=original_zooms,convert=1,linewidth=1)

        
#     #     bounding_boxes.append(bb_coordinates)
        
    
#     # Lets rescale it back!
#     new_zooms = original_zooms
#     old_zooms = (2,2,2)
#     ctd_list = RescaleCentroids(new_zooms, old_zooms, ctd_list, restrictions)
#     #Lets plot!
#     # show_centroids_new_dim1(data_original,10,ctd_list,subject)

        
#     for i, centroid in enumerate(ctd_list):

#         # if i == 4:
#         # Set the center coordinates of the sphere
#         x, y, z= centroid
#         x = np.round(x).astype(int)
#         y = np.round(y).astype(int)
#         z = np.round(z+5/original_zooms[2]).astype(int)
        
#         gap = 3
        
#         offset_x = int(np.round(gap/original_zooms[0]))
#         offset_y = int(np.round(gap/original_zooms[1]))
#         offset_z = int(np.round(gap/original_zooms[2]))

#         coordinates = (x-offset_x, x+offset_x, y-offset_y, y+offset_y, z-offset_z, z+offset_z)

#         # print(subject)
        
#         show_boundingbox_dim1(data_original, coordinates,subject,zooms=original_zooms,convert=1,linewidth=1)
#         # show_boundingbox_dim2(data_original, coordinates,subject,zooms=original_zooms,convert=0,linewidth=1)
#         # show_boundingbox_dim3(data_original, coordinates,subject,zooms=original_zooms,convert=0,linewidth=1)
    
#         l1_hu_predict = Calculate_HU_from_BoundingBox(data_original, coordinates) #Median value
        
#         #Get HU value from meta_datafile
#         temp = subject.split("_") #Split into substrings
#         subject_without_ending = "_".join(temp[:-1]) #Subject name without SERIES. So we have VERTEBRAE_LOWHU_0100_SERIES0018 -> VERTEBRAE_LOWHU_0100
        
#         filtered_df = df[df['ID'] == subject_without_ending]
#         l1_hu_target = filtered_df['ct_vertebra_l1_hu'].values[0]
        
#         #Append lists
#         subject_list.append(subject_without_ending)
#         l1_hu_target_list.append(l1_hu_target)
#         l1_hu_predict_list.append(l1_hu_predict)
        
#         print(subject_without_ending,l1_hu_target,l1_hu_predict)

# # # Create the DataFrame using the lists
# df_HU = pd.DataFrame({'Subject': subject_list,
#             'Target L1 HU': l1_hu_target_list,
#             'Prediction L1 HU': l1_hu_predict_list})

# # df_HU.to_excel('HU_output.xlsx', index=False)

#         # Set the diameter of the sphere
#         # diameter = 15

#         # # Call the function to calculate the mean value within the sphere
#         # mean_value, mask = calculate_mean_within_sphere(data_original, center_x, center_y, center_z, diameter)
        
#         # msk_data = mask*1
        
#         # show_mask_img_dim1(data_original[210:260,:,:], msk_data[210:260,:,:], subject, no_slices=40)
    