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
import nibabel.orientations as nio
#My own documents
from my_plotting_functions import *
from Load_data import *
from data_utilities import *
from my_data_utils import *
#Import networks
from SpineLocalisationNet import SpineLocalisationNet
# from new_VertebraeLocalisationNet import VertebraeLocalisationNet
from new_VertebraeLocalisationNet_batchnormdropout import VertebraeLocalisationNet
from VertebraeSegmentationNet import VertebraeSegmentationNet
#from VertebraeSegmentationNet_batchnormdropout import VertebraeSegmentationNet

#######################################################
#################### CONTROL PANEL ####################
#######################################################
#Define scans
all_scans = 0 #SET TO ZERO!! We want to process list_of_subject!!!!!!!!
list_of_subjects = ['VERTEBRAE_FRACTURE_0319_SERIES0022'] #['VERTEBRAE_FRACTURE_0224_SERIES0000','VERTEBRAE_FRACTURE_0249_SERIES0013','VERTEBRAE_FRACTURE_0261_SERIES0017','VERTEBRAE_FRACTURE_0305_SERIES0021','VERTEBRAE_FRACTURE_0319_SERIES0022','VERTEBRAE_FRACTURE_0326_SERIES0017'] #['VERTEBRAE_FRACTURE_0215_SERIES0005'] #['VERTEBRAE_LOWHU_0138_SERIES0026','VERTEBRAE_LOWHU_0191_SERIES0003',['VERTEBRAE_FRACTURE_0224_SERIES0000','VERTEBRAE_FRACTURE_0249_SERIES0013','VERTEBRAE_FRACTURE_0261_SERIES0017','VERTEBRAE_FRACTURE_0305_SERIES0021','VERTEBRAE_FRACTURE_0319_SERIES0022','VERTEBRAE_FRACTURE_0326_SERIES0017'] #['VERTEBRAE_FRACTURE_0284_SERIES0024'] #'VERTEBRAE_FRACTURE_0294_SERIES0019'] #['VERTEBRAE_FRACTURE_0294_SERIES0019'] #['VERTEBRAE_FRACTURE_0334_SERIES0010'] #['VERTEBRAE_FRACTURE_0208_SERIES0007'] #['VERTEBRAE_FRACTURE_0213_SERIES0008','VERTEBRAE_HEALTHY_0000_SERIES0003','VERTEBRAE_FRACTURE_0208_SERIES0007'] #VERTEBRAE_FRACTURE_0239_SERIES0003'] #['VERTEBRAE_HEALTHY_0001_SERIES0010'] #List of subjects
with open("E:\s174197\Thesis\My_code\Other_scripts\list_of_subjects_FRACTURE", "rb") as fp:   # Unpickling, list_of_subjects_FRACTURE lidt forkert!
    list_of_subjects = pickle.load(fp)
# with open("E:\s174197\Thesis\My_code\Other_scripts\list_of_subjects_LOWHU", "rb") as fp:   # Unpickling
#     list_of_subjects = pickle.load(fp)

# VERTEBRAE_FRACTURE_0285_SERIES0015 går den her lidt galt i SpineLocalisation?

#Define directories
dir_data_original = r'G:\DTU-Vertebra-1\NIFTI'
dir_data_stage1 = r'E:\s174197\data_RH\SpineLocalisation\data_prep\img' #'/zhome/bb/f/127616/Documents/Thesis/Rawdata_training'
dir_data_stage2 = r'E:\s174197\data_RH\VertebraeLocalisation2\data_prep\img' #'/zhome/bb/f/127616/Documents/Thesis/Rawdata_training'
dir_data_stage3 = r'E:\s174197\data_RH\VertebraeSegmentation\data_prep\img' #'/zhome/bb/f/127616/Documents/Thesis/Rawdata_training'
checkpoint_dir_stage1 = r'E:\s174197\Checkpoints\SpineLocalisation\First_try2950_batchsize1_lr0.0001_wd0.0005.pth' #First_try2650_batchsize1_lr0.0001_wd0.0005.pth' #r'E:\Andreas_s174197\Thesis\My_code\My_networks\Spine_Localisation\Checkpoints\checkpoint_batchsize1_lr0.0001_wd0.0005.pth'
# checkpoint_dir_stage2 = r'E:\s174197\Checkpoints\VertebraeLocalisation2\Batchnorm_dropout_batchsize1_lr1e-05_wd0.0001.pth' #First_try_epoch430_batchsize1_lr1e-05_wd0.0001.pth'#r'E:\s174197\Checkpoints\VertebraeLocalisation2\no_tanh_no_init_batchsize1_lr1e-05_wd0.0001.pth' #r'E:\s174197\Checkpoints\VertebraeLocalisation2\First_try_epoch430_batchsize1_lr1e-05_wd0.0001.pth' #r'E:\Andreas_s174197\Thesis\My_code\My_networks\Spine_Localisation\Checkpoints\checkpoint_batchsize1_lr0.0001_wd0.0005.pth'
checkpoint_dir_stage2 = r'E:\s174197\Checkpoints\VertebraeLocalisation2\Only_elastic_earlystopping_epoch1400_batchsize1_lr1e-05_wd0.0001.pth' #Only_elastic_earlystopping_epoch1400_batchsize1_lr1e-05_wd0.0001.pth' #First_try_epoch430_batchsize1_lr1e-05_wd0.0001.pth'
checkpoint_dir_stage3 = r'E:\s174197\Checkpoints\VertebraeSegmentation\only_elastic2_again_step33650_batchsize1_lr1e-05_wd0.0005.pth' #FIXED_DATAAUG_rotation_step13250_batchsize1_lr1e-05_wd0.0005.pth' #FIXED_DATAAUG_rotation_step11950_batchsize1_lr1e-05_wd0.0005.pth' #FIXED_DATAAUG_rotation_step12300_batchsize1_lr1e-05_wd0.0005.pth' #only_elastic2_again_step33650_batchsize1_lr1e-05_wd0.0005.pth' #FIXED_DATAAUG_rotation_step12300_batchsize1_lr1e-05_wd0.0005.pth' #only_elastic2_again_step33650_batchsize1_lr1e-05_wd0.0005.pth' #Only_rotation_batchsize1_lr1e-05_wd0.0005.pth' #First_try_step8450_batchsize1_lr1e-05_wd0.0005.pth'#Only_rotation_batchsize1_lr1e-05_wd0.0005.pth
predictions_folder = r'E:\s174197\data_RH\Predictions'

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



#Define list of scans
if all_scans:
    all_subjects = []
    for filename in listdir(dir_data_stage1):
        subject = filename.split("-")[0]
        if subject.find('FRACTURE') != -1: #PLOTTER KUN VERSE. IKKE GL
            all_subjects.append(subject)
    all_subjects = np.unique(all_subjects)
    #Sorterer fil '.DS' fra
    all_subjects = all_subjects[all_subjects != '.DS']
else:
    all_subjects = list_of_subjects
    
#Create outputfolders if they dont exist
dir_segmentationsbefore = os.path.join(predictions_folder,'Segmentations_beforeCCA_new')
dir_segmentationsafter = os.path.join(predictions_folder,'Segmentations_afterCCA_new')
dir_centroids = os.path.join(predictions_folder,'Centroids_new')
if not os.path.exists(dir_segmentationsbefore):
    os.makedirs(dir_segmentationsbefore)
if not os.path.exists(dir_segmentationsafter):
    os.makedirs(dir_segmentationsafter)
if not os.path.exists(dir_centroids):
    os.makedirs(dir_centroids)

    
# goon = False

for subject in tqdm(all_subjects):
    
    print('"'+subject+'": ')
    # continue

    # if subject == 'VERTEBRAE_LOWHU_0138_SERIES0026':
    #     goon = True
    
    # if goon == False:
    #     continue
    
    #Load original image
    img_nib_original = nib.load(os.path.join(dir_data_original,subject+'.nii.gz'))
    data_original = np.asanyarray(img_nib_original.dataobj, dtype=np.float32) #Maybe remove?
    # show_slices_dim1(data_original,subject,convert = 1, no_slices=20)
    # break
    
    
    # continue

    #Define zooms
    original_zooms = img_nib_original.header.get_zooms()
    original_shape = img_nib_original.header.get_data_shape()
    
    #Stage 1
    filename_img = subject + "-img.nii.gz"
    img_nib = nib.load(os.path.join(dir_data_stage1,filename_img))
    data_img = np.asanyarray(img_nib.dataobj, dtype=np.float32)
    
    # show_slices_dim1(data_img,subject,convert = 0, no_slices=20)
    # continue
    
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
    # show_heatmap_dim1(predictions,subject)


    #Normalise
    predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min())

    
    # show_heatmap_dim1(predictions,subject)
    
    
    
    bb_coordinates, COM = BoundingBox(predictions,restrictions)
    
    #Normal format
    data_img = data_img.squeeze(0,1).cpu().detach().numpy()

    # show_slices_dim1(data_img,no_slices=40)
    # show_heatmap_img_dim1(data_img,predictions,subject,no_slices=40)
    # show_boundingbox_dim1(data_img,bb_coordinates, subject,convert = 0,no_slices=40)
    # show_boundingbox_dim2(data_img,bb_coordinates,subject,convert = 0)
    # show_boundingbox_dim3(data_img,bb_coordinates,subject,convert = 0)
    
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

    #Load data for stage 2
    filename_img = subject + "-img.nii.gz"
    img_nib = nib.load(os.path.join(dir_data_stage2,filename_img))
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
    
    
    
    #Predictions stage 2
    #Load checkpoint
    checkpoint = torch.load(checkpoint_dir_stage2,map_location=device)

    #Define model
    model = VertebraeLocalisationNet(0.0)
    
    #Do predictions (everything else regarding sending to GPU and such is handled inside the function)
    predictions = prepare_and_predict_VLN2(data_img, model, checkpoint)

    # show_heatmap_img_dim1(data_img,predictions,subject,alpha=0.1)
    # show_heatmap_img_dim2(data_img,predictions,subject,alpha=0.1)
    
    ctd_list = find_centroids2(predictions)
    
    # show_centroids_new_dim1(data_img,ctd_list,subject,markersize=2,no_slices=30)
    
    #Get mapping
    list_of_visible_vertebrae = mapping_RH[subject]
    
    #Convert to verse format from mapping
    ctd_list = centroids_to_verse(ctd_list,list_of_visible_vertebrae)

    # show_centroids_dim1(data_img,ctd_list,subject,text=1,markersize=2,no_slices=2)
    # show_centroids_RH_toreport_dim1(data_img,ctd_list,subject,text=0,markersize=2,no_slices=40)
    # show_centroids_RH_toreport_dim2(data_img,ctd_list,subject,text=1,markersize=2,no_slices=40)
    # continue

    
    
    #Stage 3
    #Load data
    filename_img = subject + "-img.nii.gz"
    img_nib = nib.load(os.path.join(dir_data_stage3,filename_img))
    data_img = np.asanyarray(img_nib.dataobj, dtype=np.float32)
    dim1,dim2,dim3 = data_img.shape
    
    #Rescale centroids
    new_zooms = original_zooms
    old_zooms = (2,2,2)
    ctd_list = RescaleCentroids_verse(new_zooms, old_zooms, ctd_list, restrictions)
    #Rescale to (1,1,1)
    old_zooms = original_zooms
    new_zooms = (1,1,1)
    ctd_list = RescaleCentroids_verse(new_zooms, old_zooms, ctd_list)
    
    # show_centroids_new_dim1(data_img,(2,2,2),ctd_list,subject)
    # show_centroids_new_dim1(img_nib_original.get_fdata(),(2,2,2),ctd_list,subject)
    # show_centroids_new_dim2(img_nib_original.get_fdata(),(2,2,2),ctd_list,subject)
    # show_centroids_new_dim3(img_nib_original.get_fdata(),(2,2,2),ctd_list,subject)
    
    
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
    # maybecorrect_affine = np.array([[vs[0],0,0,0],[0,vs[1],0,0],[0,0,vs[2],0],[0,0,0,1]])
    # new_affine = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

    

    idx = 1
    full_output_image = np.zeros((dim1,dim2,dim3))
    full_output_list = []
    v_number_list = []
    for ctd in ctd_list[1:]:
        v_number = int(ctd[0])
        v_name = number_to_name[v_number]
        print(v_name)
        centroid = np.array([ctd[1],ctd[2],ctd[3]])
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
        # show_slices_dim1(data_img_cropped,v_name)
        # show_heatmap_dim1(heatmap,v_name)
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
        v_number_list.append(v_number)
            
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
        final_prediction[np.logical_and(mask, max_values == full_output_list[i])] = v_number_list[i]
    
    # show_mask_dim1(final_prediction, str(idx))

    #Save segmentation
    msk_nib_pred = nib.Nifti1Image(final_prediction, img_nib.affine) #new_affine) #random_affine) #img_nib.affine) #DEFINE ANOTHER AFFINE!
    nib.save(msk_nib_pred, os.path.join(dir_segmentationsbefore,subject+'-PREDICTIONbefore.nii.gz'))


    no_vertebrae = len(ctd_list)-1 #Fordi vi også har ('L','A','S') her

    for i in range(no_vertebrae):
        v_number = int(ctd_list[i+1][0])
        final_prediction_temp = deepcopy(final_prediction)
        final_prediction_temp = np.where(final_prediction_temp==v_number,1,0)

        blobs = cc3d.connected_components(final_prediction_temp,connectivity=6) #26, 18, and 6 (3D) are allowed
        no_unique = len(np.unique(blobs))

        for label in range(no_unique):
            area = np.count_nonzero(blobs == label)
            # print(area)
            if area < 10000:
                final_prediction[blobs==label] = 0



    # #Save segmentation
    # msk_nib_pred = nib.Nifti1Image(final_prediction, img_nib.affine) #new_affine) #random_affine) #img_nib.affine) #DEFINE ANOTHER AFFINE!
    # nib.save(msk_nib_pred, os.path.join(dir_segmentationsafter,subject+'-PREDICTIONafter.nii.gz'))
    # #Save centroids
    # with open(os.path.join(dir_centroids,subject+'-centroids'), "wb") as fp:   #Pickling
    #     pickle.dump(ctd_list, fp)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # #Save full segmentation
    # msk_nib = nib.Nifti1Image(final_prediction.astype(np.int16), new_affine) #img_nib.affine) #random_affine) #img_nib.affine) #DEFINE ANOTHER AFFINE!
    # nib.save(msk_nib, os.path.join(predictions_folder, subject+'-FULLSEGMENTATION.nii.gz'))
    
    
    
    
    # #Save full image with new affine
    # nifti_image = nib.Nifti1Image(img_nib, random_affine) #img_nib.affine) #random_affine) #img_nib.affine) #DEFINE ANOTHER AFFINE!
    # nib.save(nifti_image, os.path.join(predictions_folder, subject+'_FULLIMG.nii.gz'))
    
    
    #RESAMPLE AND REORIENT BACK
    # #Original zooms and orientation
    # vs = original_zooms
    # axs_code = nio.ornt2axcodes(nio.io_orientation(img_nib_original.affine)) #Image orientation
    # #image
    # # img_resampled = resample_nib(nifti_image, voxel_spacing=vs, order=3)
    # # img_resampled_reoriented = reorient_to(img_resampled, axcodes_to=axs_code)
    # #Mask
    # msk_resampled = resample_nib(msk_nib, voxel_spacing=vs, order=0) # or resample based on img: resample_mask_to(msk_nib, img_iso)
    # msk_resampled_reoriented = reorient_to(msk_resampled, axcodes_to=axs_code)
    
    # affine = np.array([[vs[0],0,0,0],[0,vs[1],0,0],[0,0,vs[2],0],[0,0,0,1]])
    
    # # random_affine[0,0] = vs[0]
    # # random_affine[1,1] = vs[1]
    # # random_affine[2,2] = vs[2]

    
    # #Save as nifti with random affine
    # img_nib = nib.Nifti1Image(img_nib_original.dataobj, affine) #img_nib.affine) #random_affine) #img_nib.affine) #DEFINE ANOTHER AFFINE!
    # msk_nib = nib.Nifti1Image(msk_resampled_reoriented.dataobj, affine) #img_nib.affine) #random_affine) #img_nib.affine) #DEFINE ANOTHER AFFINE!

    # nib.save(img_nib, os.path.join(predictions_folder, subject+'_image.nii.gz'))
    # nib.save(msk_nib, os.path.join(predictions_folder, subject+'_mask.nii.gz'))

    
    #Save old 
    














#Før havde jeg defineret den her. Men den bliver vist ikke rigtig brugt
# padding_specifications_dir = 'E:/s174197/data_RH/SpineLocalisation/Padding_specifications/pad'





















    # #Save segmentation
    # msk_nib_pred = nib.Nifti1Image(final_prediction, img_nib.affine) #new_affine) #random_affine) #img_nib.affine) #DEFINE ANOTHER AFFINE!
    # nib.save(msk_nib_pred, os.path.join(dir_FULL_SEGMENTATIONS_before,subject+'_PREDICTIONbefore.nii.gz'))


    # #CLEAN FOR SMALL ISOLATED BLOBS
    # mask = final_prediction
    
    # binary_img = np.where(mask > 0,1,0)
    # blobs = cc3d.connected_components(binary_img,connectivity=6) #26, 18, and 6 (3D) are allowed
    
    # no_unique= len(np.unique(blobs))
    
    # # ## STOP HERE AND PLOT IN DEBUG MODE TO SEE BLOBS
    # # for i in range(blobs.shape[0]):
    # #     fig, ax = plt.subplots()
    # #     ax.imshow(blobs[i,:,:].T,cmap='jet',vmin=0, vmax=no_unique-1,origin="lower") #or cmap = 'viridis'
    # #     plt.show()
    
    # for label in range(no_unique):
    #     area = np.count_nonzero(blobs == label)
    #     # print(area)
    #     if area < 1000:
    #         blobs[blobs==label] = 0
            
    # mask[blobs==0] = 0
    
    # # show_mask_dim1(mask,subject)
    # # show_mask_img_dim1(data_img,mask,subject)

    # no_vertebrae = len(ctd_list)-1 #Fordi vi også har ('L','A','S') her

    # #Save segmentation
    # msk_nib = nib.Nifti1Image(mask, img_nib.affine) #new_affine) #img_nib.affine) #random_affine) #img_nib.affine) #DEFINE ANOTHER AFFINE!
    # nib.save(msk_nib, os.path.join(dir_segmentations,subject+'-FULLSEGMENTATION.nii.gz'))
    # #Save centroids
    # np.save(os.path.join(dir_centroids,subject+'-centroids.npy'), ctd_list)
    
    
    