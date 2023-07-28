# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 21:50:12 2023

@author: PC
"""

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
#My own documents
from my_plotting_functions import *
from my_data_utils import *
from data_utilities import *
#Import networks
from SpineLocalisationNet import SpineLocalisationNet
# from new_VertebraeLocalisationNet import VertebraeLocalisationNet
from new_VertebraeLocalisationNet_batchnormdropout import VertebraeLocalisationNet
from VertebraeSegmentationNet import VertebraeSegmentationNet
# from VertebraeSegmentationNet_batchnormdropout import VertebraeSegmentationNet

#######################################################
#################### CONTROL PANEL ####################
#######################################################
#Define scans
all_scans = 0
list_of_subjects = ['sub-verse502']#['VERTEBRAE_FRACTURE_0213_SERIES0008','VERTEBRAE_HEALTHY_0000_SERIES0003','VERTEBRAE_FRACTURE_0208_SERIES0007'] #VERTEBRAE_FRACTURE_0239_SERIES0003'] #['VERTEBRAE_HEALTHY_0001_SERIES0010'] #List of subjects


# VERTEBRAE_FRACTURE_0285_SERIES0015 går den her lidt galt i SpineLocalisation?

#Define directories
dir_data_original = r'E:\s174197\data_Verse\Verse20_test_unpacked'
dir_data_stage1 = r'E:\s174197\data_Verse\SpineLocalisation\Verse20_test_prep_NOPADDING'
dir_data_stage2 = r'E:\s174197\data_Verse\VertebraeLocalisation2\Verse20_test_prep_NOPADDING'
dir_data_stage3 = r'E:\s174197\data_Verse\VertebraeSegmentation\Verse20_test_prep_NOPADDING'
# dir_padding_specifications_stage1 = r'E:\s174197\data_Verse\SpineLocalisation\Padding_specifications\pad_test'
# dir_padding_specifications_stage2 =  r'E:\s174197\data_Verse\VertebraeLocalisation2\Padding_specifications\pad_test'
# dir_padding_specifications_stage3 = r'E:\s174197\data_Verse\VertebraeSegmentation\Padding_specifications\pad_test'
checkpoint_dir_stage1 = r'E:\s174197\Checkpoints\SpineLocalisation\First_try2650_batchsize1_lr0.0001_wd0.0005.pth' #r'E:\Andreas_s174197\Thesis\My_code\My_networks\Spine_Localisation\Checkpoints\checkpoint_batchsize1_lr0.0001_wd0.0005.pth'
checkpoint_dir_stage2 = r'E:\s174197\Checkpoints\VertebraeLocalisation2\Batchnorm_dropout_batchsize1_lr1e-05_wd0.0001.pth' #First_try_epoch430_batchsize1_lr1e-05_wd0.0001.pth'#r'E:\s174197\Checkpoints\VertebraeLocalisation2\no_tanh_no_init_batchsize1_lr1e-05_wd0.0001.pth' #r'E:\s174197\Checkpoints\VertebraeLocalisation2\First_try_epoch430_batchsize1_lr1e-05_wd0.0001.pth' #r'E:\Andreas_s174197\Thesis\My_code\My_networks\Spine_Localisation\Checkpoints\checkpoint_batchsize1_lr0.0001_wd0.0005.pth'
# checkpoint_dir_stage2 = r'E:\s174197\Checkpoints\VertebraeLocalisation2\First_try_epoch430_batchsize1_lr1e-05_wd0.0001.pth'
checkpoint_dir_stage3 = r'E:\s174197\Checkpoints\VertebraeSegmentation\Only_rotation_batchsize1_lr1e-05_wd0.0005.pth' #First_try_step8450_batchsize1_lr1e-05_wd0.0005.pth'#Only_rotation_batchsize1_lr1e-05_wd0.0005.pth
predictions_folder = r'E:\s174197\data_Verse\Predictions'

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


#Define rest of data folders
dir_stage1_img = os.path.join(dir_data_stage1,'img')
dir_stage1_ctd = os.path.join(dir_data_stage1,'ctd')

dir_stage2_img = os.path.join(dir_data_stage2,'img')
dir_stage2_ctd = os.path.join(dir_data_stage2,'ctd')

dir_stage3_img = os.path.join(dir_data_stage3,'img')
dir_stage3_msk = os.path.join(dir_data_stage3,'msk')
dir_stage3_heatmaps = os.path.join(dir_data_stage3,'heatmaps')


#Create outputfolders if they dont exist
dir_segmentations = os.path.join(predictions_folder,'Segmentations')
dir_centroids = os.path.join(predictions_folder,'Centroids')
if not os.path.exists(dir_segmentations):
    os.makedirs(dir_segmentations)
if not os.path.exists(dir_centroids):
    os.makedirs(dir_centroids)


# #Load padding_specifications
# #Stage1
# with open(dir_padding_specifications_stage1, 'rb') as f:
#     padding_specifications_stage1 = pickle.load(f) 
# #Stage2
# with open(dir_padding_specifications_stage2, 'rb') as f:
#     padding_specifications_stage2 = pickle.load(f) 
# #Stage3
# with open(dir_padding_specifications_stage3, 'rb') as f:
#     padding_specifications_stage3 = pickle.load(f) 
    

#Define list of scans
if all_scans:
    all_subjects = []
    for filename in listdir(dir_stage1_img):
        if filename.startswith('.'): #Vi gider ikke have punktum-filerne med...
            continue
        subject = filename.split("_")[0]
        # if subject.find('verse') != -1: #PLOTTER KUN VERSE. IKKE GL
        all_subjects.append(subject)
    all_subjects = np.unique(all_subjects)
    #Sorterer fil '.DS' fra
    all_subjects = all_subjects[all_subjects != '.DS']
    
else:
    all_subjects = list_of_subjects


for subject in tqdm(all_subjects):  

    ################ Load original image ################
    filename_img = [f for f in listdir(dir_data_original) if (f.startswith(subject) and f.endswith('img.nii.gz'))][0]
    img_nib_original = nib.load(os.path.join(dir_data_original,filename_img))
    img_nib_original = reorient_to(img_nib_original, axcodes_to=New_orientation) #REORIENT
    data_original = np.asanyarray(img_nib_original.dataobj, dtype=np.float32) #Maybe remove?
    # show_slices_dim1(data_original,subject,convert = 1, no_slices=60)

    #Define zooms
    original_zooms = img_nib_original.header.get_zooms()
    original_shape = img_nib_original.header.get_data_shape()

    
    
    
    
    
    
    
    
    
    #############################################################
    ########################## STAGE 1 ##########################
    #############################################################
    filename_img = subject + "_img.nii.gz"
    img_nib = nib.load(os.path.join(dir_stage1_img,filename_img))
    data_img = np.asanyarray(img_nib.dataobj, dtype=np.float32)
    
    
    # show_slices_dim1(data_img,subject)
        
    #Do padding
    data_img, restrictions = center_and_pad(data=data_img, new_dim=(64,64,128), pad_value=-1)







    ########################## Create GT ##########################
    # Define file names
    filename_ctd = [f for f in listdir(dir_stage1_ctd) if (f.startswith(subject) and f.endswith('json'))][0]
    # Load files
    ctd_list_GT = load_centroids(os.path.join(os.path.join(dir_stage1_ctd,filename_ctd)))
    
    #Apply transformation to centroids
    for i in range(len(ctd_list_GT)-1):
        ctd_list_GT[i+1][1] += restrictions[0]
        ctd_list_GT[i+1][2] += restrictions[2]
        ctd_list_GT[i+1][3] += restrictions[4]
    
    #Sigma
    sigma = 3
    
    #Get data shape
    dim1, dim2, dim3 =  data_img.shape

    heatmap = np.zeros((dim1,dim2,dim3)) #UNCOMMENT IF YOU WANT SUM
    
    #Calculate heatmap
    for v in ctd_list_GT[1:]:
        new_heatmap = gaussian_kernel_3d_new(origins = (v[1],v[2],v[3]), meshgrid_dim = (dim1,dim2,dim3), gamma=1, sigma = 3)
        new_heatmap = (new_heatmap - new_heatmap.min()) / (new_heatmap.max() - new_heatmap.min())
        heatmap += new_heatmap #UNCOMMENT IF YOU WANT SUM
        #heatmaps_list.append(new_heatmap) #UNCOMMENT IF YOU WANT MAXMIMUM
    
    #heatmap = reduce(np.maximum,heatmaps_list) #UNCOMMENT IF YOU WANT MAXIMUM *heatmaps_list
                
    #Normalise heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    
    #Thresholding
    heatmap[heatmap < 0.001] = 0
    
    # show_heatmap_img_dim1(data_img,heatmap,subject)
    #############################################################


    
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
    #Normalise
    predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min())

    
    # show_heatmap_dim1(predictions,subject)
    
    
    
    bb_coordinates, COM = BoundingBox(predictions,restrictions)
    bb_coordinates_GT, COM = BoundingBox(heatmap,restrictions)
    
    #Normal format
    data_img = data_img.squeeze(0,1).cpu().detach().numpy()

    # show_slices_dim1(data_img,no_slices=40)
    # show_heatmap_img_dim1(data_img,predictions,subject,no_slices=40)
    # show_boundingbox_dim1(data_img,bb_coordinates, subject,convert = 0,no_slices=40)
    # show_boundingbox_dim2(data_img,bb_coordinates,subject,convert = 0)
    # show_boundingbox_dim3(data_img,bb_coordinates,subject,convert = 0)
    # show_boundingboxes_dim2(data_img, bb_coordinates, bb_coordinates_GT, subject)

    

    #Define zooms
    old_zooms = (8,8,8)
    new_zooms = original_zooms
    #Rescale bounding box
    original_bb_coordinates, original_COM = RescaleBoundingBox(new_zooms,old_zooms,bb_coordinates,COM,restrictions,borders=original_shape)
    
    #Make sure that bounding box coordinates goes all the way up and down on the z-axis
    # original_bb_coordinates[4] = 0
    # original_bb_coordinates[5] = data_original.shape[2]
    
    # show_boundingbox_dim1(data_original,original_bb_coordinates,subject,convert=1,zooms=original_zooms,no_slices=100)

    #Rescale again
    new_zooms = (2,2,2)
    old_zooms = original_zooms
    new_bb_coordinates, new_COM = RescaleBoundingBox(new_zooms,old_zooms,original_bb_coordinates,original_COM)













    #############################################################
    ########################## STAGE 2 ##########################
    #############################################################
    #Load data for stage 2
    filename_img = subject + "_img.nii.gz"
    img_nib = nib.load(os.path.join(dir_stage2_img,filename_img))
    data_img = np.asanyarray(img_nib.dataobj, dtype=np.float32)
    
    #Shape
    dim1, dim2, dim3 = data_img.shape
    
    #Crop from BOUNDING BOX
    x_min, x_max, y_min, y_max, z_min, z_max = new_bb_coordinates
    x_min = max(0,np.round(x_min).astype(int))
    x_max = min(np.round(x_max).astype(int),dim1-1)
    y_min = max(0,np.round(y_min).astype(int))
    y_max = min(np.round(y_max).astype(int),dim2-1)
    z_min = max(0,np.round(z_min).astype(int))
    z_max = min(np.round(z_max).astype(int),dim3-1)
    x_range = [x_min,x_max]
    y_range = [y_min,y_max]
    z_range = [z_min,z_max]
    data_img = data_img[x_range[0]:x_range[1]+1,y_range[0]:y_range[1]+1,z_range[0]:z_range[1]+1]
    
    #Crop from COM
    # new_x_COM, new_y_COM, new_z_COM = new_COM
    # new_x_COM = np.round(new_x_COM).astype(int)
    # new_y_COM = np.round(new_y_COM).astype(int)
    # new_z_COM = np.round(new_z_COM).astype(int)
    # dim1, dim2, dim3 = data_img.shape
    # x_range = [max(new_x_COM-48,0),min(new_x_COM+48,dim1)]
    # y_range = [max(new_y_COM-48,0),min(new_y_COM+48,dim2)]
    # z_range = [0,0]
    # data_img = data_img[x_range[0]:x_range[1],y_range[0]:y_range[1],:]

    #Do padding
    data_img, restrictions = center_and_pad(data=data_img, new_dim=(96,96,128), pad_value=-1)
    #Update restrictions for above cropping
    restrictions = tuple(restrictions - np.array([x_min,x_max,y_min,y_max,z_min,z_max]))
    
    # show_slices_dim1(data_img,subject)
    # continue
    
    
    
    #Predictions stage 2
    #Load checkpoint
    checkpoint = torch.load(checkpoint_dir_stage2,map_location=device)

    #Define model
    model = VertebraeLocalisationNet(0.0)
    
    #Do predictions (everything else regarding sending to GPU and such is handled inside the function)
    predictions = prepare_and_predict_VLN2(data_img, model, checkpoint)

    show_heatmap_img_dim1(data_img,predictions,subject,alpha=0.2)
    
    break
    ctd_list = find_centroids(predictions)
    
    show_centroids_new_dim1(data_img,ctd_list,subject,markersize=2,no_slices=1000)
    
    continue
    
    
    #Stage 3
    #Load data
    filename_img = subject + "_img.nii.gz"
    img_nib = nib.load(os.path.join(dir_stage3_img,filename_img))
    data_img = np.asanyarray(img_nib.dataobj, dtype=np.float32)
    dim1,dim2,dim3 = data_img.shape
    
    #Rescale centroids
    new_zooms = original_zooms
    old_zooms = (2,2,2)
    ctd_list = RescaleCentroids(new_zooms, old_zooms, ctd_list, restrictions)
    #Rescale to (1,1,1)
    old_zooms = original_zooms
    new_zooms = (1,1,1)
    ctd_list = RescaleCentroids(new_zooms, old_zooms, ctd_list)
    
    # show_centroids_new_dim1(data_img,ctd_list,subject,markersize=5)
    # show_centroids_new_dim1(img_nib_original.get_fdata(),(2,2,2),ctd_list,subject)
    # show_centroids_new_dim2(img_nib_original.get_fdata(),(2,2,2),ctd_list,subject)
    # show_centroids_new_dim3(img_nib_original.get_fdata(),(2,2,2),ctd_list,subject)
    # continue
    
    
    #Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #device = torch.device('cpu')
    checkpoint = torch.load(checkpoint_dir_stage3,map_location=device)

    #Define model
    model = VertebraeSegmentationNet(0.0)
    #Send to GPU
    model.to(device)
    # Load the saved weights into the model
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # random_affine = np.array([[-2.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    #           1.98454294e+02],
    #         [-0.00000000e+00,  2.00000000e+00,  0.00000000e+00,
    #         -1.98326700e+02],
    #         [ 0.00000000e+00, -0.00000000e+00,  2.00000000e+00,
    #           1.85275000e+03],
    #         [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    #           1.00000000e+00]])
    
    # random_affine = np.array([[1, 0, 0, -78],
    #                           [0, 1, 0, -76],
    #                           [0, 0, 1, -64],
    #                           [0, 0, 0, 1]])
    
    vs = original_zooms
    maybecorrect_affine = np.array([[vs[0],0,0,0],[0,vs[1],0,0],[0,0,vs[2],0],[0,0,0,1]])
    new_affine = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

    

    idx = 1
    full_output_image = np.zeros((dim1,dim2,dim3))
    full_output_list = []
    for centroid in ctd_list:
        full_output_temp = np.zeros((dim1,dim2,dim3))
        data_img_cropped, restrictions = center_and_pad(data=data_img, new_dim=(128,128,96), pad_value=-1,centroid=tuple(centroid))
        #Change centroid coordinate based on above cropping and padding
        x_min_restrict, _, y_min_restrict, _, z_min_restrict, _ = restrictions #OBS. x_max, y_max og z_max kan kun bruges ved bounding box!! Man skal tage trække padding og cropping fra først og SÅ kan man lægge dimensionen til. Der bliver en parantes fejl ellers.
        centroid = centroid + np.array([x_min_restrict,y_min_restrict,z_min_restrict]) #PLUS, because we are applying changes. Not reverting.

        #Create heatmap
        heatmap = gaussian_kernel_3d_new(origins=tuple(centroid), meshgrid_dim=(128,128,96), gamma=1, sigma=5)
        #Normalize
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        #Thresholding
        heatmap[heatmap < 0.001] = 0
        
        #print("Plot")
        # show_heatmap_img_dim1(data_img_cropped, heatmap, str(idx))
        
        #Prepare data for network 3


        #Input tensor
        inputs = np.stack((data_img_cropped, heatmap), axis=3)
        #Reshape
        inputs = np.moveaxis(inputs, -1, 0)
        inputs = inputs.astype(np.float32)
        
        #Save image of vertebrae
        # nifti_image = nib.Nifti1Image(inputs[0], new_affine)
        # nib.save(nifti_image, os.path.join(predictions_folder, subject+'_V'+str(idx)+'_IMG.nii.gz'))

        
        inputs = torch.from_numpy(inputs)
        inputs = inputs.unsqueeze(0) 
        
        #Send to device
        inputs = inputs.to(device)
        
        #Define sigmoid function
        sigmoid = nn.Sigmoid()
        
        # Set the model to evaluation mode
        model.eval() 
        with torch.no_grad():
            output = model(inputs)
            #Apply sigmoid
            predictions = sigmoid(output)
            #Set to 1 and 0.
            # predictions = torch.where(predictions > 0.5, torch.tensor(1), torch.tensor(0))
            
        #Change formatting
        predictions = predictions.squeeze()
        predictions = predictions.cpu().detach().numpy()
        
        #print("Plot")
        # show_mask_dim1(predictions, str(idx))
        # show_mask_img_dim1(data_img_cropped,predictions, str(idx))
        
        #Save each vertebra
        # nifti_image = nib.Nifti1Image(predictions.astype(np.float32), new_affine) #img_nib.affine)# random_affine) #img_nib.affine) #DEFINE ANOTHER AFFINE!
        # nib.save(nifti_image, os.path.join(predictions_folder, subject+'_V'+str(idx)+'.nii.gz'))
        
        ranges = [-x_min_restrict,-x_min_restrict+127,-y_min_restrict,-y_min_restrict+127,-z_min_restrict,-z_min_restrict+95] #MINUS TO EVERYTHING because we are going backwards. Not applying padding or cropping but reverting!

        
        limX1 = 0
        limX2 = 127
        limY1 = 0
        limY2 = 127
        limZ1 = 0
        limZ2 = 95
        
        if ranges[0] < 0:
            #Define offset for cropping mask
            limX1 = abs(ranges[0])
            #Set indices for full image
            ranges[0] = 0
                        
        if ranges[1] > dim1-1:
            #Define offset for cropping mask
            cut_off = ranges[1]-(dim1-1)
            limX2 = 127 - cut_off
            #Set indices for full image
            ranges[1] = dim1-1

        if ranges[2] < 0:
            #Define offset for cropping mask
            limY1 = abs(ranges[2])
            #Set indices for full image
            ranges[2] = 0
 
        if ranges[3] > dim2-1:
            #Define offset for cropping mask
            cut_off = ranges[3]-(dim2-1)
            limY2 = 127 - cut_off
            #Set indices for full image
            ranges[3] = dim2-1
            
        if ranges[4] < 0:
            #Define offset for cropping mask
            limZ1 = abs(ranges[4])
            #Set indices for full image
            ranges[4] = 0
            
        if ranges[5] > dim3-1:
            #Define offset for cropping mask
            cut_off = ranges[5]-(dim3-1)
            limZ2 = 95 - cut_off
            #Set indices for full image
            ranges[5] = dim3-1

        #Crop prediction
        predictions = predictions[limX1:limX2+1,limY1:limY2+1,limZ1:limZ2+1]
        full_output_temp[ranges[0]:ranges[1]+1,ranges[2]:ranges[3]+1,ranges[4]:ranges[5]+1] = predictions #Minus because we are transfering back!
        
        # show_mask_dim1(full_output_temp, str(idx))

        
        full_output_list.append(full_output_temp)

            
        idx+=1
    
    # Convert the list of arrays into a single NumPy array
    full_output_list = np.array(full_output_list)
    
    # Find the maximum values along the specified axis (axis=0 for elementwise comparison)
    max_values = np.max(full_output_list, axis=0)
    
    # Create a mask indicating where the maximum values are above 0.5
    mask = max_values > 0.5
    
    # Initialize an output array with zeros
    final_prediction = np.zeros_like(max_values)
    
    # Assign values based on the conditions
    for i in range(len(full_output_list)):
        final_prediction[np.logical_and(mask, max_values == full_output_list[i])] = i + 1
    
    # show_mask_dim1(final_prediction, str(idx))





    #CLEAN FOR SMALL ISOLATED BLOBS
    mask = final_prediction
    
    binary_img = np.where(mask > 0,1,0)
    blobs = cc3d.connected_components(binary_img,connectivity=6) #26, 18, and 6 (3D) are allowed
    
    no_unique= len(np.unique(blobs))
    
    # ## STOP HERE AND PLOT IN DEBUG MODE TO SEE BLOBS
    # for i in range(blobs.shape[0]):
    #     fig, ax = plt.subplots()
    #     ax.imshow(blobs[i,:,:].T,cmap='jet',vmin=0, vmax=no_unique-1,origin="lower") #or cmap = 'viridis'
    #     plt.show()
    
    for label in range(no_unique):
        area = np.count_nonzero(blobs == label)
        # print(area)
        if area < 200:
            blobs[blobs==label] = 0
            
    mask[blobs==0] = 0
    
    # show_mask_dim1(mask,subject)
    
    #Save segmentation
    msk_nib = nib.Nifti1Image(mask, img_nib.affine) #new_affine) #random_affine) #img_nib.affine) #DEFINE ANOTHER AFFINE!
    nib.save(msk_nib, os.path.join(dir_segmentations,subject+'-FULLSEGMENTATION.nii.gz'))
    #Save centroids
    np.save(os.path.join(dir_centroids,subject+'-centroids.npy'), ctd_list)



















    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #device = torch.device('cpu')
    # checkpoint = torch.load(checkpoint_dir_stage1,map_location=device)

    # #Define model
    # model = SpineLocalisationNet()
    # #Send to GPU
    # model.to(device)
    # # Load the saved weights into the model
    # model.load_state_dict(checkpoint['model_state_dict'])
    
    # data_img = torch.from_numpy(data_img)
    # data_img = data_img.unsqueeze(0) 
    # data_img = data_img.unsqueeze(0)
    
    # #send to device
    # data_img = data_img.to(device)
        
    # # Set the model to evaluation mode
    # model.eval() 
    # with torch.no_grad():
    #     predictions = model(data_img)
        
    # #Change formatting
    # predictions = predictions.squeeze()
    # predictions = predictions.cpu().detach().numpy()
    # #Normalise
    # predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min())
    
    # show_heatmap_dim1(predictions,subject)
    
    # #Load restrictions
    # restrictions = padding_specifications_stage1[subject]
    
    # bb_coordinates, COM = BoundingBox(predictions,restrictions)
    
    # #Normal format
    # data_img = data_img.squeeze(0,1).cpu().detach().numpy()

    # # show_slices_dim1(data_img,no_slices=40)
    # # show_heatmap_img_dim1(data_img,predictions,subject,no_slices=40)
    # # show_boundingbox_dim1(data_img,bb_coordinates, subject,convert = 0,no_slices=40)
    # # show_boundingbox_dim2(data_img,bb_coordinates,subject,convert = 0)
    # # show_boundingbox_dim3(data_img,bb_coordinates,subject,convert = 0)
    
    # #Define zooms
    # old_zooms = (8,8,8)
    # new_zooms = original_zooms
    # #Rescale bounding box
    # original_bb_coordinates, original_COM = RescaleBoundingBox(new_zooms,old_zooms,bb_coordinates,COM,restrictions,borders=data_original.shape)
    
    # #Make sure that bounding box coordinates goes all the way up and down on the z-axis
    # # original_bb_coordinates[4] = 0
    # # original_bb_coordinates[5] = data_original.shape[2]
    
    # show_boundingbox_dim1(data_original,original_bb_coordinates,subject,convert=1,zooms=original_zooms,no_slices=100)

    # #Rescale again
    # new_zooms = (2,2,2)
    # old_zooms = original_zooms
    # new_bb_coordinates, new_COM = RescaleBoundingBox(new_zooms,old_zooms,original_bb_coordinates,original_COM)

    # #Load data for stage 2
    # filename_img = subject + "_img.nii.gz"
    # img_nib = nib.load(os.path.join(dir_stage2_img,filename_img))
    # data_img = np.asanyarray(img_nib.dataobj, dtype=np.float32)
    
    # #Crop from BOUNDING BOX
    # x_min, x_max, y_min, y_max, z_min, z_max = new_bb_coordinates
    # x_min = np.round(x_min).astype(int)
    # x_max = np.round(x_max).astype(int)
    # y_min = np.round(y_min).astype(int)
    # y_max = np.round(y_max).astype(int)
    # z_min = np.round(z_min).astype(int)
    # z_max = np.round(z_max).astype(int)
    # x_range = [x_min,x_max]
    # y_range = [y_min,y_max]
    # z_range = [z_min,z_max]
    # data_img = data_img[x_range[0]:x_range[1]+1,y_range[0]:y_range[1]+1,z_range[0]:z_range[1]+1]
    
    # #Crop from COM
    # # new_x_COM, new_y_COM, new_z_COM = new_COM
    # # new_x_COM = np.round(new_x_COM).astype(int)
    # # new_y_COM = np.round(new_y_COM).astype(int)
    # # new_z_COM = np.round(new_z_COM).astype(int)
    # # dim1, dim2, dim3 = data_img.shape
    # # x_range = [max(new_x_COM-48,0),min(new_x_COM+48,dim1)]
    # # y_range = [max(new_y_COM-48,0),min(new_y_COM+48,dim2)]
    # # z_range = [0,0]
    # # data_img = data_img[x_range[0]:x_range[1],y_range[0]:y_range[1],:]

    # #Do padding
    # data_img, restrictions = center_and_pad(data=data_img, new_dim=(96,96,128), pad_value=-1)
    # #Update restrictions for above cropping
    # restrictions = tuple(restrictions - np.array([x_min,x_max,y_min,y_max,z_min,z_max]))
    
    # show_slices_dim1(data_img,subject)
    # break
    
    
    
    # #Predictions stage 2 (+10 er helt fint. + 20 er lidt meget...)
    # #Load checkpoint
    # checkpoint = torch.load(checkpoint_dir_stage2,map_location=device)

    # #Define model
    # model = VertebraeLocalisationNet(0.0)
    
    # #Do predictions (everything else regarding sending to GPU and such is handled inside the function)
    # predictions = prepare_and_predict_VLN2(data_img, model, checkpoint)

    # # show_heatmap_img_dim1(data_img,predictions,subject,alpha=0.2)
    
    # ctd_list = find_centroids(predictions)
    
    # # show_centroids_new_dim1(data_img,(3,3,3),ctd_list,subject)    
    
    
    # #Stage 3
    # #Load data
    # filename_img = subject + "-img.nii.gz"
    # img_nib = nib.load(os.path.join(dir_data_stage3,filename_img))
    # data_img = np.asanyarray(img_nib.dataobj, dtype=np.float32)
    # dim1,dim2,dim3 = data_img.shape
    
    # #Rescale centroids
    # new_zooms = original_zooms
    # old_zooms = (2,2,2)
    # ctd_list = RescaleCentroids(new_zooms, old_zooms, ctd_list, restrictions)
    # #Rescale to (1,1,1)
    # old_zooms = original_zooms
    # new_zooms = (1,1,1)
    # ctd_list = RescaleCentroids(new_zooms, old_zooms, ctd_list)
    
    # # show_centroids_new_dim1(data_img,(2,2,2),ctd_list,subject)
    # # show_centroids_new_dim1(img_nib_original.get_fdata(),(2,2,2),ctd_list,subject)
    # # show_centroids_new_dim2(img_nib_original.get_fdata(),(2,2,2),ctd_list,subject)
    # # show_centroids_new_dim3(img_nib_original.get_fdata(),(2,2,2),ctd_list,subject)
    
    
    # #Load model
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #device = torch.device('cpu')
    # checkpoint = torch.load(checkpoint_dir_stage3,map_location=device)

    # #Define model
    # model = VertebraeSegmentationNet(0.0)
    # #Send to GPU
    # model.to(device)
    # # Load the saved weights into the model
    # model.load_state_dict(checkpoint['model_state_dict'])
    
    # random_affine = np.array([[-2.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    #           1.98454294e+02],
    #         [-0.00000000e+00,  2.00000000e+00,  0.00000000e+00,
    #         -1.98326700e+02],
    #         [ 0.00000000e+00, -0.00000000e+00,  2.00000000e+00,
    #           1.85275000e+03],
    #         [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    #           1.00000000e+00]])
    

    # idx = 1
    # full_output_image = np.zeros((dim1,dim2,dim3))
    # full_output_list = []
    # for centroid in ctd_list:
    #     full_output_temp = np.zeros((dim1,dim2,dim3))
    #     data_img_cropped, restrictions = center_and_pad(data=data_img, new_dim=(128,128,96), pad_value=-1,centroid=tuple(centroid))
    #     #Change centroid coordinate based on above cropping and padding
    #     x_min_restrict, _, y_min_restrict, _, z_min_restrict, _ = restrictions #OBS. x_max, y_max og z_max kan kun bruges ved bounding box!! Man skal tage trække padding og cropping fra først og SÅ kan man lægge dimensionen til. Der bliver en parantes fejl ellers.
    #     centroid = centroid + np.array([x_min_restrict,y_min_restrict,z_min_restrict]) #PLUS, because we are applying changes. Not reverting.

    #     #Create heatmap
    #     heatmap = gaussian_kernel_3d_new(origins=tuple(centroid), meshgrid_dim=(128,128,96), gamma=1, sigma=5)
    #     #Normalize
    #     heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
    #     #Thresholding
    #     heatmap[heatmap < 0.001] = 0
        
    #     #print("Plot")
    #     # show_heatmap_img_dim1(data_img_cropped, heatmap, str(idx))
        
    #     #Prepare data for network 3


    #     #Input tensor
    #     inputs = np.stack((data_img_cropped, heatmap), axis=3)
    #     #Reshape
    #     inputs = np.moveaxis(inputs, -1, 0)
    #     inputs = inputs.astype(np.float32)
        
    #     inputs = torch.from_numpy(inputs)
    #     inputs = inputs.unsqueeze(0) 
        
    #     #Send to device
    #     inputs = inputs.to(device)
        
    #     #Define sigmoid function
    #     sigmoid = nn.Sigmoid()
        
    #     # Set the model to evaluation mode
    #     model.eval() 
    #     with torch.no_grad():
    #         output = model(inputs)
    #         #Apply sigmoid
    #         predictions = sigmoid(output)
    #         #Set to 1 and 0.
    #         predictions = torch.where(predictions > 0.5, torch.tensor(1), torch.tensor(0))
            
    #     #Change formatting
    #     predictions = predictions.squeeze()
    #     predictions = predictions.cpu().detach().numpy()
        
    #     #print("Plot")
    #     # show_mask_dim1(predictions, str(idx))
        
    #     #Save each vertebra
    #     # nifti_image = nib.Nifti1Image(predictions.astype(np.float32), img_nib.affine) #DEFINE ANOTHER AFFINE!
    #     # nib.save(nifti_image, os.path.join(predictions_folder, subject+'_V'+str(idx)+'.nii.gz'))
        
    #     ranges = [-x_min_restrict,-x_min_restrict+127,-y_min_restrict,-y_min_restrict+127,-z_min_restrict,-z_min_restrict+95] #MINUS TO EVERYTHING because we are going backwards. Not applying padding or cropping but reverting!

        
    #     limX1 = 0
    #     limX2 = 127
    #     limY1 = 0
    #     limY2 = 127
    #     limZ1 = 0
    #     limZ2 = 95
        
    #     if ranges[0] < 0:
    #         #Define offset for cropping mask
    #         limX1 = abs(ranges[0])
    #         #Set indices for full image
    #         ranges[0] = 0
                        
    #     if ranges[1] > dim1-1:
    #         #Define offset for cropping mask
    #         cut_off = ranges[1]-(dim1-1)
    #         limX2 = 127 - cut_off
    #         #Set indices for full image
    #         ranges[1] = dim1-1

    #     if ranges[2] < 0:
    #         #Define offset for cropping mask
    #         limY1 = abs(ranges[2])
    #         #Set indices for full image
    #         ranges[2] = 0
 
    #     if ranges[3] > dim2-1:
    #         #Define offset for cropping mask
    #         cut_off = ranges[3]-(dim2-1)
    #         limY2 = 127 - cut_off
    #         #Set indices for full image
    #         ranges[3] = dim2-1
            
    #     if ranges[4] < 0:
    #         #Define offset for cropping mask
    #         limZ1 = abs(ranges[4])
    #         #Set indices for full image
    #         ranges[4] = 0
            
    #     if ranges[5] > dim3-1:
    #         #Define offset for cropping mask
    #         cut_off = ranges[5]-(dim3-1)
    #         limZ2 = 95 - cut_off
    #         #Set indices for full image
    #         ranges[5] = dim3-1

    #     #Crop prediction
    #     predictions = predictions[limX1:limX2+1,limY1:limY2+1,limZ1:limZ2+1]
    #     full_output_temp[ranges[0]:ranges[1]+1,ranges[2]:ranges[3]+1,ranges[4]:ranges[5]+1] = predictions #Minus because we are transfering back!
        
    #     # show_mask_dim1(full_output_temp, str(idx))

        
    #     full_output_list.append(full_output_temp)

            
    #     idx+=1
    
    # # Convert the list of arrays into a single NumPy array
    # full_output_list = np.array(full_output_list)
    
    # # Find the maximum values along the specified axis (axis=0 for elementwise comparison)
    # max_values = np.max(full_output_list, axis=0)
    
    # # Create a mask indicating where the maximum values are above 0.5
    # mask = max_values > 0.5
    
    # # Initialize an output array with zeros
    # final_prediction = np.zeros_like(max_values)
    
    # # Assign values based on the conditions
    # for i in range(len(full_output_list)):
    #     final_prediction[np.logical_and(mask, max_values == full_output_list[i])] = i + 1
    
            
    # # #Save full image
    # nifti_image = nib.Nifti1Image(final_prediction.astype(np.int16), img_nib.affine) #DEFINE ANOTHER AFFINE!
    # nib.save(nifti_image, os.path.join(predictions_folder, subject+'_FULL.nii.gz'))














#Før havde jeg defineret den her. Men den bliver vist ikke rigtig brugt
# padding_specifications_dir = 'E:/s174197/data_RH/SpineLocalisation/Padding_specifications/pad'
