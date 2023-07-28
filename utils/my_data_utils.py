import torch
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from os import listdir
import nibabel
import plotly.graph_objects as go
import numpy as np
import matplotlib as mpl
import cc3d
from time import sleep
import math
import torch.nn as nn
from copy import deepcopy
import cc3d
from functools import reduce 
import cv2 #For finding contours
from scipy.spatial.distance import cdist #For calculating euclidian distance
from my_plotting_functions import *


#Dictionaries mapping
number_to_name = {
    1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7',
    8: 'T1', 9: 'T2', 10: 'T3', 11: 'T4', 12: 'T5', 13: 'T6', 14: 'T7',
    15: 'T8', 16: 'T9', 17: 'T10', 18: 'T11', 19: 'T12', 20: 'L1',
    21: 'L2', 22: 'L3', 23: 'L4', 24: 'L5', 25: 'L6', 26: 'Sacrum',
    27: 'Cocc', 28: 'T13'
}

name_to_number = {
    'C1': 1, 
    'C2': 2, 
    'C3': 3, 
    'C4': 4, 
    'C5': 5, 
    'C6': 6, 
    'C7': 7, 
    'T1': 8,
    'T2': 9,
    'T3': 10,
    'T4': 11,
    'T5': 12,
    'T6': 13,
    'T7': 14,
    'T8': 15,
    'T9': 16,
    'T10': 17,
    'T11': 18,
    'T12': 19,
    'L1': 20,
    'L2': 21,
    'L3': 22,
    'L4': 23,
    'L5': 24,
    'L6': 25,
    'Sacrum': 26,
    'Cocc': 27,
    'T13': 28
}




def Predict(dataloader, model, checkpoint_dir, output_dir):
    """
    Predicts data for SpineLocalisationNet and saves it

    Arguments:
    dataloader - a Pytorch dataloader with data to predict
    model - the architecture of the network network defined as a pytorch model. For instance UNet3D()
    checkpoint_dir - the directory for the file containing the checkpoint of the model
    output_dir - the directory for where to save the output. Will create the folder if it does not exits.
    """
    
    if not os.path.exists(output_dir):
               os.makedirs(output_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #device = torch.device('cpu')
    checkpoint = torch.load(checkpoint_dir,map_location=device)

    #Define model
    model = model
    #Send to GPU ?? - det var ikke tilføjet før..
    model.to(device)
    # Load the saved weights into the model
    model.load_state_dict(checkpoint['model_state_dict'])
    
    
    # Set the model to evaluation mode
    model.eval() 
    
    with torch.no_grad():
        for inputs, subject in tqdm(dataloader):
            assert len(subject) == 1 #Make sure we are only predicting one batch
            inputs = inputs.to(device)
            predictions = model(inputs)
            predictions = predictions.squeeze()
            torch.save(predictions, os.path.join(output_dir,subject[0] + '_heatmap_pred.pt'))


def Predict_VLN(dataloader, model, checkpoint_dir, output_dir):
    """
    Predicts data for VerteBraeLocalisationNet and saves it

    Arguments:
    dataloader - a Pytorch dataloader with data to predict
    model - the architecture of the network network defined as a pytorch model. For instance UNet3D()
    checkpoint_dir - the directory for the file containing the checkpoint of the model
    output_dir - the directory for where to save the output. Will create the folder if it does not exits.
    """
    
    if not os.path.exists(output_dir):
               os.makedirs(output_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #device = torch.device('cpu')
    checkpoint = torch.load(checkpoint_dir,map_location=device)

    #Define model
    model = model
    #Send to GPU!
    model.to(device)
    # Load the saved weights into the model
    model.load_state_dict(checkpoint['model_state_dict'])
    
    
    # Set the model to evaluation mode
    model.eval() 
    
    with torch.no_grad():
        for i, (img, _, inputs_list, targets_list, start_end_voxels, subject) in enumerate(tqdm(dataloader)):
            assert len(subject) == 1 #Make sure we are only predicting one batch
            _, dim1, dim2, dim3 = img.shape
            outputs_list = []
            for i in range(len(inputs_list)):
                #Initialise vector
                full_output_temp = torch.empty((1,8,dim1,dim2,dim3))
                #Unpack targets and inputs and get predictions (I know I don't need the targets)
                inputs = inputs_list[i]
                targets = targets_list[i]
                #Sent to devicexe
                inputs, targets = inputs.to(device), targets.to(device)
                #Forward pass
                output = model(inputs)
                #Get start and end voxel
                start_voxel = start_end_voxels[i][0].item()
                end_voxel = start_end_voxels[i][1].item()
                #Put into the output_tensor
                full_output_temp[:,:,:,:,start_voxel:end_voxel+1] = output
                #Append to list
                outputs_list.append(full_output_temp)
            #Taking the maximum response
            prediction = torch.max(*outputs_list)
            #Save
            torch.save(prediction, os.path.join(output_dir,subject[0] + '_heatmap_pred.pt'))



def Predict_VSN(dataloader, model, checkpoint_dir, output_dir):
    """
    Predicts data for VerteSegmentationNet and saves it

    Arguments:
    dataloader - a Pytorch dataloader with data to predict
    model - the architecture of the network network defined as a pytorch model. For instance UNet3D()
    checkpoint_dir - the directory for the file containing the checkpoint of the model
    output_dir - the directory for where to save the output. Will create the folder if it does not exits.
    """
    
    if not os.path.exists(output_dir):
               os.makedirs(output_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #device = torch.device('cpu')
    checkpoint = torch.load(checkpoint_dir,map_location=device)

    #Define model
    model = model
    #Send to GPU ??
    model.to(device)
    # Load the saved weights into the model
    model.load_state_dict(checkpoint['model_state_dict'])
    
    #Define sigmoid function
    sigmoid = nn.Sigmoid()

    # Compute accuracies on validation set.
    model.eval() #Set to evaluation
    with torch.no_grad():
        for inputs, targets, subject in tqdm(dataloader):
            # if subject[0] == 'sub-verse599-23':
            #     print("Now")
            # continue
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            #Apply sigmoid
            predictions = sigmoid(output)
            #Set to 1 and 0.
            predictions = torch.where(predictions > 0.5, torch.tensor(1), torch.tensor(0))

            torch.save(predictions, os.path.join(output_dir,subject[0] + '_msk_pred.pt'))


def Plot_loss(checkpoint_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #device = torch.device('cpu')
    checkpoint = torch.load(checkpoint_dir,map_location=device)
    train_loss = checkpoint['train_loss']
    val_loss = checkpoint['val_loss']

    #Old code
    # plt.figure()
    # plt.plot(train_loss, label='Training',linewidth=1)
    # plt.plot(val_loss, label='Validation',linewidth=1)
    # plt.title(r'Training-curve')
    # plt.xlabel(r'Epochs')
    # plt.ylabel(r'MSE-loss') #r'$\mathcal{L} ( \mathbf{x} )$'
    # plt.legend(loc='upper right')
    # plt.show()
    
    #Reset to standard setting
    #mpl.rcParams.update(mpl.rcParamsDefault)
        
    #New code
    # set style
    
    #plt.rcParams['figure.figsize'] = np.array([6.4, 4.8])
    #plt.rcParams['font.size'] = 20

    # set style
    plt.style.use('seaborn-darkgrid')
    
    # create figure and axes
    fig, ax = plt.subplots() #Default is fig_size=(6.4, 4.8). Call like this plt.subplots(fig_size=(6.4, 4.8))
    
    # plot training and validation loss
    ax.plot(train_loss, label='Training', linewidth=2)
    ax.plot(val_loss, label='Validation', linewidth=2)
    
    # set title and axis labels
    #ax.set_title('Training Curve',fontsize=16)
    ax.set_xlabel('Epochs',fontsize=16)
    ax.set_ylabel('MSE Loss',fontsize=16)
    
    # add legend
    leg = ax.legend(loc='upper right',fontsize=16)
    for line in leg.get_lines(): #Define linewidth
        line.set_linewidth(1.0)
        
    # set intervals for x and y axes
    #ax.set_xlim([0, 2000])
    #ax.set_ylim([0, max(max(train_loss), max(val_loss))])
    
    # display grid
    ax.grid(True)
    
    #Save
    # plot_name = 'SpineLocalisation_trainingcurve.png'
    # plt.savefig('/Users/andreasaspe/Library/Mobile Documents/com~apple~CloudDocs/DTU/12.semester/Thesis/Figures/'+plot_name, dpi=300)

    # # show plot
    # plt.show()
    

def BoundingBox(heatmap_data,restrictions=None):
    """
    This function is designed to create a bounding box around the heatmap of the whole spine.
    
    Inputs
    heatmap_data: The heatmap data as an array
    restrictions: Optional argument, if the bounding box should be adjusted for going
                  outside of padding boundaries. Can be necessary on a downsampled image.
    
    Outputs
    coordinates: The min and max coordinate for the bounding box in all three dimensions,
                  given like this: x_min, x_max, y_min, y_max, z_min, z_max
    COM:         The 3D center of mass coordinates, given as a tuple: (x,y,z)
    """
    
    dim1,dim2,dim3 = heatmap_data.shape
    x,y,z = np.meshgrid(np.arange(dim1),np.arange(dim2),np.arange(dim3),indexing='ij')
    x_COM = sum(sum(sum(heatmap_data*x)))/sum(sum(sum(heatmap_data)))
    y_COM = sum(sum(sum(heatmap_data*y)))/sum(sum(sum(heatmap_data)))
    z_COM = sum(sum(sum(heatmap_data*z)))/sum(sum(sum(heatmap_data)))
    
    x_lower = int(x_COM)-7
    x_upper = int(x_COM)+7
    y_lower = int(y_COM)-7 #5?
    y_upper = int(y_COM)+7
    # z_lower = int(z_COM)-20
    # z_upper = int(z_COM)+20
    #img_data = img_data[x_lower:x_upper , : , :]
    heatmap_data = heatmap_data[x_lower:x_upper , y_lower:y_upper  , :] #z_lower:z_upper
    
    binary_img = np.where(heatmap_data > 0.3,1,0)
    blobs = cc3d.connected_components(binary_img,connectivity=6) #26, 18, and 6 (3D) are allowed

    no_unique= len(np.unique(blobs))
    
    # STOP HERE AND PLOT IN DEBUG MODE TO SEE BLOBS
    # for i in range(blobs.shape[0]):
    #     fig, ax = plt.subplots()
    #     ax.imshow(blobs[i,:,:].T,cmap='jet',vmin=0, vmax=no_unique-1,origin="lower") #or cmap = 'viridis'
    #     plt.show()

    for label in range(no_unique):
        area = np.count_nonzero(blobs == label)
        if area < 200: #Måske 200?
            blobs[blobs==label] = 0
        # else:
        #     print(area)
    
    # for i in range(blobs.shape[0]):
    #     fig, ax = plt.subplots()
    #     ax.imshow(blobs[i,:,:].T,cmap='jet',vmin=0, vmax=no_unique-1,origin="lower") #or cmap = 'viridis'
    #     plt.show()

    indices = np.where(blobs > 0)
            
    x_min, y_min, z_min = min(indices[0]), min(indices[1]), min(indices[2])
    x_max, y_max, z_max = max(indices[0]), max(indices[1]), max(indices[2])
    
    #Tager højde for at have croppet billede og derfor er x_min kun et relativt koordinat.
    #Tager max og min for at sætte det til hhv. 0 eller dimensionen af aksen, hvis bounding box koordinat lægger udenfor billedet.
    x_min = max(x_lower + x_min - 3,0) #2?
    x_max = min(x_lower + x_max + 3,dim1)
    y_min = max(y_lower + y_min - 7,0) #5?
    y_max = min(y_lower + y_max + 3,dim2) #2... MEN ender med 3. #3 for verse, men mere for RH
    z_min = max(z_min - 5,0)
    z_max = dim3 #min(z_max + 5,dim3) #3? #ALL THE WAY UP on z-axis!
    
    # x_min = x_lower + x_min - 2
    # x_max = x_lower + x_max + 3xxwxw
    # y_min = y_lower + y_min - 5
    # y_max = y_lower + y_max + 1
    # z_min = z_min - 7
    # z_max = z_max + 3

    #Check if bounding box is outside of padding
    if restrictions!=None:
        #Unpack tuple (x_min_restrict: Lowest possible voxel value. x_max restrixt: Highest possible voxel value)
        x_min_restrict, x_max_restrict, y_min_restrict, y_max_restrict, z_min_restrict, z_max_restrict = restrictions

        x_min = max(x_min,x_min_restrict)
        x_max = min(x_max,x_max_restrict)
        y_min = max(y_min,y_min_restrict)
        y_max = min(y_max,y_max_restrict)
        z_min = max(z_min,z_min_restrict)
        z_max = min(z_max,z_max_restrict)

    coordinates = (x_min, x_max, y_min, y_max, z_min, z_max)
    COM = (x_COM, y_COM, z_COM)

    return coordinates, COM

    


def BoundingBoxFromCOM(heatmap_data,restrictions=None):
    """
    This function is designed to create a bounding box around the heatmap of the whole spine FROM COM!
    
    Inputs
    heatmap_data: The heatmap data as an array
    restrictions: Optional argument, if the bounding box should be adjusted for going
                  outside of padding boundaries. Can be necessary on a downsampled image.
    
    Outputs
    coordinates: The min and max coordinate for the bounding box in all three dimensions,
                  given like this: x_min, x_max, y_min, y_max, z_min, z_max
    COM:         The 3D center of mass coordinates, given as a tuple: (x,y,z)
    """
    
    dim1,dim2,dim3 = heatmap_data.shape
    x,y,z = np.meshgrid(np.arange(dim1),np.arange(dim2),np.arange(dim3),indexing='ij')
    x_COM = sum(sum(sum(heatmap_data*x)))/sum(sum(sum(heatmap_data)))
    y_COM = sum(sum(sum(heatmap_data*y)))/sum(sum(sum(heatmap_data)))
    z_COM = sum(sum(sum(heatmap_data*z)))/sum(sum(sum(heatmap_data)))
    
    COM = (x_COM, y_COM, z_COM)
    
    x_COM_rounded = np.round(x_COM).astype(int)
    y_COM_rounded = np.round(y_COM).astype(int)
    z_COM_rounded = np.round(z_COM).astype(int)
    
    x_min = x_COM_rounded-12
    x_max = x_COM_rounded+1
    y_min = y_COM_rounded-12
    y_max = y_COM_rounded+11
    z_min = 0
    z_max = dim3
    
    #Check if bounding box is outside of padding
    if restrictions!=None:
        #Unpack tuple (x_min_restrict: Lowest possible voxel value. x_max restrixt: Highest possible voxel value)
        x_min_restrict, x_max_restrict, y_min_restrict, y_max_restrict, z_min_restrict, z_max_restrict = restrictions

        x_min = max(x_min,x_min_restrict)
        x_max = min(x_max,x_max_restrict)
        y_min = max(y_min,y_min_restrict)
        y_max = min(y_max,y_max_restrict)
        z_min = max(z_min,z_min_restrict)
        z_max = min(z_max,z_max_restrict)

    coordinates = (x_min, x_max, y_min, y_max, z_min, z_max)

    return coordinates, COM


def RescaleBoundingBox(new_zooms,old_zooms,old_coordinates,COM=None,restrictions = None, borders = None):
    """
    This function can rescale the Bounding Box to any other resolution.
    
    Arguments
    new_zooms:    tuple containing the voxel size in mm for the NEW image.
    old_zooms:    tuple containing the voxel size in mm for the OLD image.
    coordinates:  The coordinates of the old bounding box as a tuple. This is the output of the function BoundingBox
                  The format is (x_min, x_max, y_min, y_max, z_min, z_max)
    COM:          The 3D coordinates for Center of Mass, given as a tuple - (x,y,z)
    restrictions: Optional argument, if the bounding box should change origin. This is necessary if you have padding.
    borders:      Optional argument. This is in order to check if it goes outside the border of new image. Format is data.shape (dim1,dim2,dim3)
    
    Returns
    rescaled_coordinates: The RESCALED min and max coordinate for the bounding box in all three dimensions,
                          given like this: x_min, x_max, y_min, y_max, z_min, z_max
    rescaled_COM:         The rescaled center of mass coordinates, given as a tuple: (x,y,z)
    """

    #Coordinates unpacking
    x_min, x_max, y_min, y_max, z_min, z_max = old_coordinates
    if COM != None:
        x_COM, y_COM, z_COM = COM
    
    #Necessary if you have been using padding!
    if restrictions != None:
        #Unpack tuple
        x_min_restrict, x_max_restrict, y_min_restrict, y_max_restrict, z_min_restrict, z_max_restrict = restrictions

        x_min = x_min - x_min_restrict
        x_max = x_max - x_min_restrict  #or x_min + (x_max-x_ min). Dvs. så længder man længden til. Men det er det samme som at trække forskubningen pga. padding fra.
        y_min = y_min - y_min_restrict 
        y_max = y_max - y_min_restrict 
        z_min = z_min - z_min_restrict 
        z_max = z_max - z_min_restrict

        #COM
        if COM != None:
            x_COM = x_COM - x_min_restrict
            y_COM = y_COM - y_min_restrict
            z_COM = z_COM - z_min_restrict


    #New voxel coordinates. Remember that x_min etc. is old voxel coordinates.
    new_x_min = (x_min)*old_zooms[0]/new_zooms[0] #Add minus one for slack! DELETED BECAUSE IT GOT TOO BIG...
    new_x_max = (x_max)*old_zooms[0]/new_zooms[0] #Add plus one for slack! DELETED BECAUSE IT GOT TOO BIG...

    new_y_min = (y_min)*old_zooms[1]/new_zooms[1] #Add minus one for slack! DELETED BECAUSE IT GOT TOO BIG...
    new_y_max = (y_max)*old_zooms[1]/new_zooms[1] #Add plus one for slack! DELETED BECAUSE IT GOT TOO BIG...

    new_z_min = (z_min-1)*old_zooms[2]/new_zooms[2] #Add minus one for slack!
    new_z_max = (z_max+1)*old_zooms[2]/new_zooms[2] #Add plus one for slack!

    #COM
    if COM != None:
        new_x_COM = x_COM*old_zooms[0]/new_zooms[0]
        new_y_COM = y_COM*old_zooms[1]/new_zooms[1]
        new_z_COM = z_COM*old_zooms[2]/new_zooms[2]
    
    if borders != None:
        dim1,dim2,dim3=borders
        
        new_x_min = max(0,new_x_min)
        new_x_max = min(new_x_max,dim1-1)
        new_y_min = max(0,new_y_min)
        new_y_max = min(new_y_max,dim2-1)
        new_z_min = max(0,new_z_min)
        new_z_max = min(new_z_max,dim3-1)

    rescaled_coordinates = (new_x_min, new_x_max, new_y_min, new_y_max, new_z_min, new_z_max)
    if COM != None:
        rescaled_COM = (new_x_COM, new_y_COM, new_z_COM)

    if COM != None:
        return rescaled_coordinates, rescaled_COM
    else:
        return rescaled_coordinates



def Calculate_HU_from_BoundingBox(data_img,coordinates):
    x_min, x_max, y_min, y_max, z_min, z_max = coordinates
    
    #Crop volume
    median_value = np.median(data_img[x_min:x_max+1,y_min:y_max+1,z_min:z_max+1])
    
    return median_value


def dice_score(mask1, mask2):
    intersection = np.sum(mask1 * mask2)  # Calculate the intersection of the two masks
    mask1_sum = np.sum(mask1)  # Sum of pixels in mask1
    mask2_sum = np.sum(mask2)  # Sum of pixels in mask2
    dice = (2.0 * intersection) / (mask1_sum + mask2_sum + 1e-8)  # Add a small epsilon to avoid division by zero
    return dice



def RescaleCentroids(new_zooms,old_zooms,ctd_list,restrictions = None):
    """
    This function can rescale centroids for the naive own edition of ctd_list.
    
    Arguments
    new_zooms:    tuple containing the voxel size in mm for the NEW image.
    old_zooms:    tuple containing the voxel size in mm for the OLD image.
    ctd_list:     The coordinates of the old centroid coordinates box as a list of arrays. The format should be without direction as first element and no labels.
    restrictions: Optional argument, if the bounding box should change origin. This is necessary if you have been padding or cropping.
    
    
    Returns
    rescaled_coordinates: The rescaled ctd_list.
    """
    
    if restrictions != None:
        #Unpack tuple
        x_min_restrict, _, y_min_restrict, _, z_min_restrict, _ = restrictions
        #Convert (we are taking each x coordinate and subtract with x_min_restrict, each y coordinate subtracted with y_min_restrict and each z coordinate subtracted with z_min_restrict)
        ctd_list = ctd_list - np.array([x_min_restrict,y_min_restrict,z_min_restrict])

    for ctd in ctd_list:
        ctd[0] = ctd[0]*old_zooms[0]/new_zooms[0]
        ctd[1] = ctd[1]*old_zooms[1]/new_zooms[1]
        ctd[2] = ctd[2]*old_zooms[2]/new_zooms[2]
        
    return ctd_list


def RescaleCentroids_verse(new_zooms,old_zooms,ctd_list,restrictions = None):
    """
    This function can rescale centroids for the verse edition of ctd_list.
    
    Arguments
    new_zooms:    tuple containing the voxel size in mm for the NEW image.
    old_zooms:    tuple containing the voxel size in mm for the OLD image.
    ctd_list:     The coordinates of the old centroid coordinates box as a list of arrays. The format should be without direction as first element and no labels.
    restrictions: Optional argument, if the bounding box should change origin. This is necessary if you have been padding or cropping.
    
    
    Returns
    rescaled_coordinates: The rescaled ctd_list.
    """

    #Convert to array
    array = np.array(ctd_list[1:])
    
    if restrictions != None:
        #Unpack tuple
        x_min_restrict, _, y_min_restrict, _, z_min_restrict, _ = restrictions
        #Convert (we are taking each x coordinate and subtract with x_min_restrict, each y coordinate subtracted with y_min_restrict and each z coordinate subtracted with z_min_restrict)
        array[:,1:] = array[:,1:]-np.array([x_min_restrict,y_min_restrict,z_min_restrict])

    for ctd in array:
        ctd[1] = ctd[1]*old_zooms[0]/new_zooms[0]
        ctd[2] = ctd[2]*old_zooms[1]/new_zooms[1]
        ctd[3] = ctd[3]*old_zooms[2]/new_zooms[2]

    #Back to verse
    ctd_list = array.tolist()
    ctd_list.insert(0, ('L','A','S'))
        
    return ctd_list


def gaussian_kernel_3d(origins, meshgrid_dim, sigma=1):
    x,y,z = origins
    mesh_x,mesh_y,mesh_z = meshgrid_dim
    
    x_grid, y_grid, z_grid = np.meshgrid(np.arange(mesh_x), np.arange(mesh_y),np.arange(mesh_z),indexing='ij')
    kernel = np.exp(-((x_grid - x)**2 + (y_grid - y)**2 + (z_grid - z)**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel

def gaussian_kernel_3d_new(origins, meshgrid_dim, gamma, sigma=1):
    d=3 #dimension
    x,y,z = origins
    mesh_x,mesh_y,mesh_z = meshgrid_dim
    
    x_grid, y_grid, z_grid = np.meshgrid(np.arange(mesh_x), np.arange(mesh_y),np.arange(mesh_z),indexing='ij')
    kernel = np.exp(-((x_grid - x)**2 + (y_grid - y)**2 + (z_grid - z)**2) / (2 * sigma**2))
    factor = gamma/( (2*math.pi)**(d/2)*sigma**d   )
    heatmap = factor*kernel
    return heatmap


def center_and_pad(data,new_dim,pad_value,centroid=None):
    """
    This function can do center and padding of an image
    
    Arguments
    data: Image to be padded given as 3D numpy array
    new_dim: The dimensions after padding
    pad_value: The value which will be padded. In most cases it should be -1, because this step is done after preprocessing.
    centoid: The 3D coordinate which should be center of image, given as tuple or as array. The function will try to ensure this if possible by cropping
    for the variable new_dim. 
    
    Returns
    data_adjusted: The data after padding and possibly cropping.
    restrictions: A 6D array containing info about padding and cropping. The structure is (x_min, x_max, y_min, y_max, z_min, z_max).
    """
    
    dim1, dim2, dim3 = data.shape
    dim1_new, dim2_new, dim3_new = new_dim
    
    if centroid != None:
        x,y,z = centroid
        
        y = y-20
        
        x_start = int(max(np.round(x-dim1_new/2),0)) #64
        x_end = int(min(np.round(x+dim1_new/2-1),dim1-1)) #63
        y_start = int(max(np.round(y-dim2_new/2),0))
        y_end = int(min(np.round(y+dim2_new/2-1),dim2-1))
        z_start = int(max(np.round(z-dim3_new/2),0))
        z_end = int(min(np.round(z+dim3_new/2-1),dim3-1))
        
        data_adjusted = deepcopy(data[x_start:x_end+1,y_start:y_end+1,z_start:z_end+1])
    else:
        data_adjusted = deepcopy(data)
        #For opdating restrctions. x_start and such will always be zero in case of plotting bounding box I think..
        x_start = 0
        y_start = 0
        z_start = 0
    
    #Get dimensions after cropping
    dim1, dim2, dim3 = data_adjusted.shape
    
    #Calculate padding in each side (volume should be centered)
    padding_dim1 = (dim1_new-dim1)/2
    padding_dim2 = (dim2_new-dim2)/2
    padding_dim3 = (dim3_new-dim3)/2
    
    #Calculate padding in each side by taking decimal values into account
    #Dim1
    if padding_dim1 > 0:
        if np.floor(padding_dim1) == padding_dim1:
            pad1 = (int(padding_dim1),int(padding_dim1))
        else:
            pad1 = (int(np.floor(padding_dim1)),int(np.floor(padding_dim1)+1))
    else:
        pad1 = (0,0)
    #Dim2
    if padding_dim2 > 0:
        if np.floor(padding_dim2) == padding_dim2:
            pad2 = (int(padding_dim2),int(padding_dim2))
        else:
            pad2 = (int(np.floor(padding_dim2)),int(np.floor(padding_dim2)+1))
    else:
        pad2 = (0,0)
    #Dim3
    if padding_dim3 > 0:
        if np.floor(padding_dim3) == padding_dim3:
            pad3 = (int(padding_dim3),int(padding_dim3))
        else:
            pad3 = (int(np.floor(padding_dim3)),int(np.floor(padding_dim3)+1))
    else:
        pad3 = (0,0)
        
    restrictions = (pad1[0]-x_start , pad1[0]-x_start+(dim1-1)   ,   pad2[0]-y_start , pad2[0]-y_start+(dim2-1)   ,   pad3[0]-z_start , pad3[0]-z_start+(dim3-1))

    #Doing padding
    data_adjusted=np.pad(data_adjusted, (pad1, pad2, pad3), constant_values = pad_value)
    
    return data_adjusted, restrictions



def find_centroids(heatmap_data_prediction):
    binary_img = np.where(heatmap_data_prediction > 0.3,1,0)
    
    #show_mask_dim1(binary_image,subject)
    #show_heatmap_img_dim1(img_data,heatmap_data_target[:,:,:],subject,no_slices=20,alpha=0.3)
    #show_heatmap_img_dim1(img_data,heatmap_data_prediction,subject,no_slices=20,alpha=0.3)
    #show_heatmap_dim1(heatmap_data_prediction,subject,no_slices=100,alpha=0.3)
    #show_slices_dim1(img_data,subject)
        

    blobs = cc3d.connected_components(binary_img,connectivity=6)
    
    no_unique= len(np.unique(blobs))
    
    min_z_coordinates=[]
    for label in range(1,no_unique):
        min_z_coordinates.append(np.min(np.where(blobs == label)[2]))
    #print(min_z_coordinates)
    
    #Lowest_blob_index
    Lowest_blob_index = np.argmin(min_z_coordinates)+1 #Fordi jeg sortede 0 fra før oppe i forløbet. Se mit range. 0 er baggrunds-class
    
    #Get mask of lowest blob
    mask_lowest_blob = np.where(blobs == Lowest_blob_index,1,0)
    
    #Get index of maximum value
    #heatmap_cropped = heatmap_data_prediction[mask_lowest_blob == 1]
    
    # Get indices of non-zero values in the binary mask
    nonzero_indices = np.nonzero(mask_lowest_blob)
    
    # Create a list of tuples containing the coordinates
    coordinates = list(zip(nonzero_indices[0], nonzero_indices[1], nonzero_indices[2]))
    
    # Sort the coordinates based on the corresponding values in heatmap_array
    sorted_coordinates = sorted(coordinates, key=lambda coord: heatmap_data_prediction[coord], reverse=True)
    
    # # Get the coordinates of the 10 highest values
    # top_10_coordinates = sorted_coordinates[:10]
    
    # avg_coordinate = np.array([
    #     sum(coord[0] for coord in top_10_coordinates) / len(top_10_coordinates),
    #     sum(coord[1] for coord in top_10_coordinates) / len(top_10_coordinates),
    #     sum(coord[2] for coord in top_10_coordinates) / len(top_10_coordinates)
    # ])
    
    
    # Get the coordinates of the highest value
    max_coordinate = sorted_coordinates[0]
    
    max_coordinate = np.array([
        max_coordinate[0],
        max_coordinate[1],
        max_coordinate[2]
    ])
    
    coordinates_list = [max_coordinate]
    
    #Check sequence
    x_coordinate = int(np.round(max_coordinate[0])) #Current x_coordinate
    y_coordinate = int(np.round(max_coordinate[1])) #Current y_coordinate
    z_coordinate = int(np.round(max_coordinate[2])) #Current z_coordinate
    
    _, _, dim3 = heatmap_data_prediction.shape
    while abs(z_coordinate+10 - dim3-1) > 6:
    
        
        x_range = [x_coordinate-20,x_coordinate+20]
        y_range = [y_coordinate-20,y_coordinate+20]
        z_range = [z_coordinate+10,z_coordinate+20]
        offset = np.array([x_range[0],y_range[0],z_range[0]])
        
        heatmap_cropped = heatmap_data_prediction[x_range[0]:x_range[1],y_range[0]:y_range[1],z_range[0]:z_range[1]]
        
        #show_heatmap_dim1(heatmap_cropped,subject)
        # Flatten the 3D array
        flattened_array = heatmap_cropped.flatten()
        
        #If leq than 5, then it means that we are even outside the range or we have two few points to average over
        if flattened_array.shape[0] <= 5:
            break
        
        #Check if it is worth looking for
        if max(flattened_array) < 0.1: #Det her er en anden, måske mere effektiv måde at skrive max på: flattened_array[top_10_indices[-1]]. Udnytter at jeg allerede har kørt argsort
            break  
        
        # Get the indices of the 5 highest values
        top_10_indices = np.argsort(flattened_array)[-5:]
        
        # Reshape the indices to their original 3D shape
        reshaped_indices = np.unravel_index(top_10_indices, heatmap_cropped.shape)
        
        # Get the corresponding heatmap values for getting the weights
        heatmap_values = flattened_array[top_10_indices]
        
        # Calculate the weighted average coordinate
        weighted_avg_coordinate = np.array([
            np.average(reshaped_indices[0], weights=heatmap_values),
            np.average(reshaped_indices[1], weights=heatmap_values),
            np.average(reshaped_indices[2], weights=heatmap_values)
        ])
        
        #Plus offset
        weighted_avg_coordinate = weighted_avg_coordinate + offset
        
        #Append to list
        coordinates_list.append(weighted_avg_coordinate)
        
        #Update coordinatess
        x_coordinate = int(np.round(weighted_avg_coordinate[0])) #Current x_coordinate
        y_coordinate = int(np.round(weighted_avg_coordinate[1])) #Current y_coordinate
        z_coordinate = int(np.round(weighted_avg_coordinate[2])) #Current z_coordinate
        
    return coordinates_list
        # print(x_coordinate)
        # print(y_coordinate)
        # print(z_coordinate)
        
        # i+=1
        # print(i)
    
    
def find_centroids2(heatmap_data_prediction):
    """
    Almost similar to find_centroids
    """
    binary_img = np.where(heatmap_data_prediction > 0.1,1,0)
    
    #show_mask_dim1(binary_image,subject)
    #show_heatmap_img_dim1(img_data,heatmap_data_target[:,:,:],subject,no_slices=20,alpha=0.3)
    #show_heatmap_img_dim1(img_data,heatmap_data_prediction,subject,no_slices=20,alpha=0.3)
    #show_heatmap_dim1(heatmap_data_prediction,subject,no_slices=100,alpha=0.3)
    #show_slices_dim1(img_data,subject)
        
    blobs = cc3d.connected_components(binary_img,connectivity=6)
    
    
    no_unique= len(np.unique(blobs))
    
    min_z_coordinates=[]
    for label in range(1,no_unique):
        min_z_coordinates.append(np.min(np.where(blobs == label)[2]))
    #print(min_z_coordinates)
    
    
    
    sorted_indices = np.argsort(min_z_coordinates)
    # Second_lowest_blob_index = sorted_indices[1]
    Lowest_blob_index = sorted_indices[0]
    
    # if (min_z_coordinates[Second_lowest_blob_index] - min_z_coordinates[Lowest_blob_index]) <= 5:
    #     mask_lowest_blob = np.where((blobs == Lowest_blob_index+1) | (blobs == Second_lowest_blob_index+1), 1, 0)
    # else:
    
    mask_lowest_blob = np.where(blobs == Lowest_blob_index+1,1,0)


    ###### GAMMEL ######
    # # Get indices of non-zero values in the binary mask
    # nonzero_indices = np.nonzero(mask_lowest_blob)
    
    # #Get average coordinate
    # average_coordinate = np.mean(nonzero_indices, axis=1)
      
    # coordinates_list = [average_coordinate]
    
    # #Check sequence
    # x_coordinate = int(np.round(average_coordinate[0])) #Current x_coordinate
    # y_coordinate = int(np.round(average_coordinate[1])) #Current y_coordinate
    # z_coordinate = int(np.round(average_coordinate[2])) #Current z_coordinate
    ########
    
    ###### NY ######
    # heatmap_data_prediction_temp = deepcopy(heatmap_data_prediction)
    # heatmap_data_prediction_temp[~mask_lowest_blob] = 0
    # #Show masked heatmap!
    # # show_heatmap_dim1(heatmap_data_prediction_temp,'hej')
    
    # flattened_array = heatmap_data_prediction_temp.ravel()
    
    # top_10_indices = np.argsort(flattened_array)[-10:]
    # reshaped_indices = np.unravel_index(top_10_indices, heatmap_data_prediction_temp.shape)
    
    # # Get the corresponding heatmap values for getting the weights
    # heatmap_values = flattened_array[top_10_indices]
    
    # # Calculate the weighted average coordinate
    # weighted_avg_coordinate = np.array([
    #     np.average(reshaped_indices[0], weights=heatmap_values),
    #     np.average(reshaped_indices[1], weights=heatmap_values),
    #     np.average(reshaped_indices[2], weights=heatmap_values)
    # ])
    
    # coordinates_list = [weighted_avg_coordinate]
    
    # x_coordinate = int(np.round(average_coordinate[0])) #Current x_coordinate
    # y_coordinate = int(np.round(average_coordinate[1])) #Current y_coordinate
    # z_coordinate = int(np.round(average_coordinate[2])) #Current z_coordinate
    #######

    
    
    ##### NY IGEN #####
    # Get indices of non-zero values in the binary mask
    nonzero_indices = np.nonzero(mask_lowest_blob)
    
    x_min = np.min(nonzero_indices[0])
    x_max = np.max(nonzero_indices[0])
    y_min = np.min(nonzero_indices[1])
    y_max = np.max(nonzero_indices[1])
    z_min = np.min(nonzero_indices[2])
    z_max = np.max(nonzero_indices[2])
    offset = np.array([x_min,y_min,z_min])
    
    heatmap_cropped = heatmap_data_prediction[x_min:x_max+1,y_min:y_max+1,z_min:z_max+1]

    flattened_array = heatmap_cropped.flatten()
    
    # Get the indices of the 5 highest values
    top_10_indices = np.argsort(flattened_array)[-3:]
    
    # Reshape the indices to their original 3D shape
    reshaped_indices = np.unravel_index(top_10_indices, heatmap_cropped.shape)
    
    # Get the corresponding heatmap values for getting the weights
    heatmap_values = flattened_array[top_10_indices]
    
    # Calculate the weighted average coordinate
    weighted_avg_coordinate = np.array([
        np.average(reshaped_indices[0], weights=heatmap_values),
        np.average(reshaped_indices[1], weights=heatmap_values),
        np.average(reshaped_indices[2], weights=heatmap_values)
    ])
    
    #Plus offset
    weighted_avg_coordinate = weighted_avg_coordinate + offset
    
    #Append to list
    coordinates_list = [weighted_avg_coordinate]
    
    #Update coordinatess
    x_coordinate = int(np.round(weighted_avg_coordinate[0])) #Current x_coordinate
    y_coordinate = int(np.round(weighted_avg_coordinate[1])) #Current y_coordinate
    z_coordinate = int(np.round(weighted_avg_coordinate[2])) #Current z_coordinate
    #################
    
    
    _, _, dim3 = heatmap_data_prediction.shape
    while abs(z_coordinate+10 - dim3-1) > 6:
    
        
        x_range = [x_coordinate-20,x_coordinate+20]
        y_range = [y_coordinate-20,y_coordinate+20]
        z_range = [z_coordinate+10,z_coordinate+22]
        offset = np.array([x_range[0],y_range[0],z_range[0]])
        
        heatmap_cropped = heatmap_data_prediction[x_range[0]:x_range[1],y_range[0]:y_range[1],z_range[0]:z_range[1]]
        
        # show_heatmap_dim1(heatmap_cropped,'hej')
        # Flatten the 3D array
        flattened_array = heatmap_cropped.flatten()
        
        #If leq than 5, then it means that we are even outside the range or we have two few points to average over
        if flattened_array.shape[0] <= 3: #ÆNDREDE DETTE FRA 5 til 3. GØR DET NOGEN FORSKEL? I THINK NOT
            break
        
        #Check if it is worth looking for
        if max(flattened_array) < 0.05: #Det her er en anden, måske mere effektiv måde at skrive max på: flattened_array[top_10_indices[-1]]. Udnytter at jeg allerede har kørt argsort
            break    
  
        
        # Get the indices of the 5 highest values
        top_10_indices = np.argsort(flattened_array)[-3:]
        
        # Reshape the indices to their original 3D shape
        reshaped_indices = np.unravel_index(top_10_indices, heatmap_cropped.shape)
        
        # Get the corresponding heatmap values for getting the weights
        heatmap_values = flattened_array[top_10_indices]
        
        # Calculate the weighted average coordinate
        weighted_avg_coordinate = np.array([
            np.average(reshaped_indices[0], weights=heatmap_values),
            np.average(reshaped_indices[1], weights=heatmap_values),
            np.average(reshaped_indices[2], weights=heatmap_values)
        ])
        
        #Plus offset
        weighted_avg_coordinate = weighted_avg_coordinate + offset
        
        #Append to list
        coordinates_list.append(weighted_avg_coordinate)
        
        #Update coordinates
        x_coordinate = int(np.round(weighted_avg_coordinate[0])) #Current x_coordinate
        y_coordinate = int(np.round(weighted_avg_coordinate[1])) #Current y_coordinate
        z_coordinate = int(np.round(weighted_avg_coordinate[2])) #Current z_coordinate
        
        
    return coordinates_list
        # print(x_coordinate)
        # print(y_coordinate)
        # print(z_coordinate)
        
        # i+=1
        # print(i)


def find_centroids3(heatmap_data_prediction):
    """
    This method relies just on blob findings!
    """
    binary_img = np.where(heatmap_data_prediction > 0.1,1,0)
    
    #show_mask_dim1(binary_image,subject)
    #show_heatmap_img_dim1(img_data,heatmap_data_target[:,:,:],subject,no_slices=20,alpha=0.3)
    #show_heatmap_img_dim1(img_data,heatmap_data_prediction,subject,no_slices=20,alpha=0.3)
    #show_heatmap_dim1(heatmap_data_prediction,subject,no_slices=100,alpha=0.3)
    #show_slices_dim1(img_data,subject)
        
    blobs = cc3d.connected_components(binary_img,connectivity=6)
        
        
    no_unique= len(np.unique(blobs))
    
    
    
    
    # STOP HERE AND PLOT IN DEBUG MODE TO SEE BLOBS
    # for i in range(blobs.shape[0]):
    #     fig, ax = plt.subplots()
    #     ax.imshow(blobs[i,:,:].T,cmap='jet',vmin=0, vmax=no_unique-1,origin="lower") #or cmap = 'viridis'
    #     plt.show()

    # for label in range(no_unique):
    #     area = np.count_nonzero(blobs == label)
    #     if area < 200: #Måske 200?
    #         blobs[blobs==label] = 0
    #     # else:
    #     #     print(area)
    
    # for i in range(blobs.shape[0]):
    #     fig, ax = plt.subplots()
    #     ax.imshow(blobs[i,:,:].T,cmap='jet',vmin=0, vmax=no_unique-1,origin="lower") #or cmap = 'viridis'
    #     plt.show()
    
    
    
    
    min_z_coordinates=[]
    for label in range(1,no_unique):
        min_z_coordinates.append(np.min(np.where(blobs == label)[2]))
    #print(min_z_coordinates)
    
    
    
    sorted_indices = np.argsort(min_z_coordinates)+1 #Fordi jeg sortede 0 fra før!
    
    coordinates_list = []
    
    for label in sorted_indices:
        mask = blobs == label
        heatmap_values = heatmap_data_prediction[mask]
        
        heatmap_data_prediction_temp = deepcopy(heatmap_data_prediction)
        heatmap_data_prediction_temp[~mask] = 0
        #Show masked heatmap!
        # show_heatmap_dim1(heatmap_data_prediction_temp,'hej')
        
        flattened_array = heatmap_data_prediction_temp.ravel()
        
        top_10_indices = np.argsort(flattened_array)[-10:]
        reshaped_indices = np.unravel_index(top_10_indices, heatmap_data_prediction_temp.shape)
        
        # Get the corresponding heatmap values for getting the weights
        heatmap_values = flattened_array[top_10_indices]
        
        # Calculate the weighted average coordinate
        weighted_avg_coordinate = np.array([
            np.average(reshaped_indices[0], weights=heatmap_values),
            np.average(reshaped_indices[1], weights=heatmap_values),
            np.average(reshaped_indices[2], weights=heatmap_values)
        ])
                
        #Append to list
        coordinates_list.append(weighted_avg_coordinate)
        
    return coordinates_list
        

    # STOP HERE AND PLOT IN DEBUG MODE TO SEE BLOBS
    # for i in range(blobs.shape[0]):
    #     fig, ax = plt.subplots()
    #     ax.imshow(blobs[i,:,:].T,cmap='jet',vmin=0, vmax=no_unique-1,origin="lower") #or cmap = 'viridis'
    #     plt.show()
    
    

    # for label in range(no_unique):
    #     area = np.count_nonzero(blobs == label)
    #     if area < 200: #Måske 200?
    #         blobs[blobs==label] = 0
    #     # else:
    #     #     print(area)
    
    # for i in range(blobs.shape[0]):
    #     fig, ax = plt.subplots()
    #     ax.imshow(blobs[i,:,:].T,cmap='jet',vmin=0, vmax=no_unique-1,origin="lower") #or cmap = 'viridis'
    #     plt.show()




def prepare_and_predict_VLN2(data_img,model,checkpoint):
    """
    This function can prepare and predict data for VertebraeLocalisatioNet2
    
    Arguments
    data_img:   The padded scan as numpy array
    model:      The pure model definition. Everything regarding sending to device and stuff is handled inside of this function.
    checkpoint: The LOADED checkpoint of the model.
    
    Returns
    Prediction: The prediction as numpy array.
    """
    
    dim1, dim2, dim3 = data_img.shape
    
    
    if dim3 > 128:
        list_of_images = []
        start_end_voxels = [] #It is actual voxel start and end values
        #Start values
        start_voxel = 0
        finished = False
        while not finished:
            #Find end voxel and probably renew start voxel if finished
            end_voxel = start_voxel + 127
            if end_voxel + 1 > dim3:
                start_voxel = dim3-128
                end_voxel = dim3-1
                
            start_end_voxels.append((start_voxel,end_voxel))

            #Check if we should stop after this iteration
            if end_voxel == dim3 - 1: #Will be the case if finished
                finished = True
                
            #Image
            cropped_img = data_img[:,:,start_voxel:end_voxel+1]
            #Convert to tensor
            cropped_img = torch.from_numpy(cropped_img)
            #Add one dimension to image to set #channels = 1
            cropped_img = cropped_img.unsqueeze(0).unsqueeze(0) 
            #Save to list
            list_of_images.append(cropped_img)

            #Update start-boxel (with overlap of 96 voxels.)
            start_voxel = end_voxel - 95
    else: #The dimensions are exactly right and there is only one image in list. No need for cropping. It cannot be lower because we have already done padding in a previous step.
        #Image
        cropped_img = data_img
        cropped_img = torch.from_numpy(cropped_img)
        cropped_img = cropped_img.unsqueeze(0).unsqueeze(0)
        list_of_images = [cropped_img]
        #Start end voxels
        start_voxel = 0
        end_voxel = 127
        start_end_voxels = [(start_voxel,end_voxel)]
    
    
    #Do predictions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #device = torch.device('cpu')
    #Send to GPU
    model.to(device)
    # Load the saved weights into the model
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set the model to evaluation mode
    model.eval() 

    with torch.no_grad():
        inputs_list = list_of_images
        outputs_list = []
        # local_list = []
        # spatial_list = []
        #Initialise vector
        # full_local_temp = torch.ones((1,1,dim1,dim2,dim3))*(-1000)
        # full_spatial_temp = torch.ones((1,1,dim1,dim2,dim3))*(-1000)
        for j in range(len(inputs_list)):
            full_output_temp = torch.ones((1,1,dim1,dim2,dim3))*(-1000)
            #Unpack targets and inputs and get predictions (I know I don't need the targets)
            inputs = inputs_list[j]
            #Sent to device
            inputs = inputs.to(device)
            #Forward pass
            output, _, _ = model(inputs) #output, local, spatial = model(inputs)
            
            #PLOT
            # show_slices_dim1(inputs.squeeze().detach().numpy(),'')
            
            #Get start and end voxel
            start_voxel = start_end_voxels[j][0]
            end_voxel = start_end_voxels[j][1]
            
            #PLOT
            # show_heatmap_img_dim1(inputs.squeeze().cpu().detach().numpy(),output.squeeze().cpu().detach().numpy(),'',alpha=0.1)
            
            #Put into the output_tensor
            full_output_temp[:,:,:,:,start_voxel:end_voxel+1] = output
            # full_local_temp[:,:,:,:,start_voxel:end_voxel+1] = local
            # full_spatial_temp[:,:,:,:,start_voxel:end_voxel+1] = spatial
            
            #Append to list
            outputs_list.append(full_output_temp)
            # local_list.append(full_local_temp)
            # spatial_list.append(full_spatial_temp)
            
            #PLOT
            # show_heatmap_dim1(full_output_temp.squeeze().detach().numpy(),'')
            
        #Taking the maximum response
        prediction = reduce(torch.max,outputs_list)
        
    #Convert to numpy format
    prediction = prediction.squeeze() #Squeezer vist alle 1-taller på samme tid!
    prediction = prediction.cpu().detach().numpy()
    
    #PLOT
    # show_heatmap_dim1(prediction,'')
    
    return prediction
        

def MSEloss(image1,image2):
    "Own implemention of MSE loss"
    
    squared_diff = np.square(image1 - image2)
    mse = np.mean(squared_diff)
    
    return mse


def calculate_iou(box1, box2):
    # Extract coordinates for box1
    x_min1, x_max1, y_min1, y_max1, z_min1, z_max1 = box1

    # Extract coordinates for box2
    x_min2, x_max2, y_min2, y_max2, z_min2, z_max2 = box2

    # Calculate intersection coordinates
    x_min_intersect = max(x_min1, x_min2)
    y_min_intersect = max(y_min1, y_min2)
    z_min_intersect = max(z_min1, z_min2)
    x_max_intersect = min(x_max1, x_max2)
    y_max_intersect = min(y_max1, y_max2)
    z_max_intersect = min(z_max1, z_max2)

    # Calculate intersection dimensions
    inter_width = max(0, x_max_intersect - x_min_intersect)
    inter_height = max(0, y_max_intersect - y_min_intersect)
    inter_depth = max(0, z_max_intersect - z_min_intersect)

    # Calculate intersection volume
    inter_volume = inter_width * inter_height * inter_depth

    # Calculate box1 volume
    volume1 = (x_max1 - x_min1) * (y_max1 - y_min1) * (z_max1 - z_min1)

    # Calculate box2 volume
    volume2 = (x_max2 - x_min2) * (y_max2 - y_min2) * (z_max2 - z_min2)

    # Calculate union volume
    union_volume = volume1 + volume2 - inter_volume

    # Calculate IoU
    iou = inter_volume / union_volume

    return iou


def centroids_to_verse(ctd_list,mapping):
    """
    This function can convert output from find_centroids or find_centroids2 function to verse format and from mapping
    
    Arguments
    ctd_list: centroids list in New format. Output from find_centroids function! (or find_centroids2)
    mapping: A list of visible vertebrae so it knows which to assign
    Return:
    ctd_list_VERSE: As verse format
    """
    
    ctd_list_VERSE = []
    #Convert to Verse Format
    ctd_list_VERSE.append(('L','A','S'))
    for i in range(len(ctd_list)):
        v_name = mapping[i]
        if not v_name == 'NONE':
            list_temp = []
            list_temp.append(int(name_to_number[v_name]))
            list_temp.append(ctd_list[i][0])
            list_temp.append(ctd_list[i][1])
            list_temp.append(ctd_list[i][2])
            ctd_list_VERSE.append(list_temp)

    vertebrae_sorted = np.flip(ctd_list_VERSE[1:],axis=0).tolist()
    ctd_list_VERSE[1:] = vertebrae_sorted
    
    return ctd_list_VERSE

def filter_T10_to_L5(ctd_list):
    array = np.array(ctd_list[1:]) #as_array
    
    indices = array[:,0]
    
    idx = np.where((indices >= 17) & (indices <= 24))
    
    array = array[idx]
    
    #Get sorting indices
    sorted_indices = np.argsort(array[:, 0])
    
    # Sort the array based on the sorted indices
    sorted_array = array[sorted_indices]
    
    #Back to verse
    ctd_list = sorted_array.tolist()
    
    ctd_list.insert(0, ('L','A','S'))
    return ctd_list

#Calculate distances
def Calculate_distances(ctd_list_GT,ctd_list_pred):
    #As array
    array_GT = np.array(ctd_list_GT[1:])
    array_pred = np.array(ctd_list_pred[1:])
    
    distances = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]
    vertebrae_list = np.arange(17,25)
    
    for v_number in vertebrae_list:
        try:
            correct_index_GT = np.where(array_GT[:,0] == v_number)[0][0] #Fordi vi leder i ctd_list_GT[1:]. Har sorteret den første række fra
            correct_index_pred = np.where(array_pred[:,0] == v_number)[0][0]
    
            x_GT = array_GT[correct_index_GT][1]
            y_GT = array_GT[correct_index_GT][2]
            z_GT = array_GT[correct_index_GT][3]
            x_prediction = array_pred[correct_index_pred][1]
            y_prediction = array_pred[correct_index_pred][2]
            z_prediction = array_pred[correct_index_pred][3]
        
            dist = np.sqrt( (x_GT-x_prediction)**2 + (y_GT-y_prediction)**2 + (z_GT-z_prediction)**2) * 2 #Times 2, because pixel size is 2mm. Then I get the answer in mm
            distances[abs(v_number-24)] = dist #L5 er første plads i stedet for sidste
        except:
            print("Vertebrae {} does not exist".format(number_to_name[v_number]))
        
    return distances

def find_3d_contours(binary_mask):
    """
    This function can find the contours of a 3D numpy array

    Parameters
    ----------
    binary_mask : 3D numpy binary numpy array (0 and 1s)
        The binary mask

    Returns
    -------
    contours_image_final : 3D numpy binary numpy array (0 and 1s)
        The binary mask, now only as contours!
    
    """
    
    #Convert to right format
    binary_mask = binary_mask.astype(np.uint8)
    
    #####################################
    ############### DIM 1 ###############
    #####################################
    #Initialise list
    contours_image_list = []
    
    for z in range(binary_mask.shape[0]):
        # Extract the 2D slice (z-th plane)
        slice_z = binary_mask[z, :, :]

        # Find contours in the 2D slice
        contours, _ = cv2.findContours(slice_z, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a new blank image to draw the contours
        contour_image = np.zeros((binary_mask.shape[1],binary_mask.shape[2]))
        
        # Draw the contours on the new image
        cv2.drawContours(contour_image, contours, -1, 1, thickness=1)
        # show_one_slice(contour_image,subject)

        # Append the contours to the 3D contours list
        contours_image_list.append(contour_image)
            
    contours_image_final_dim1 = np.array(contours_image_list)
    # show_mask_dim1(contours_image_final_dim1,'hej')
    
    
    
    #####################################
    ############### DIM 2 ###############
    #####################################
    #Initialise list
    contours_image_list = []
    
    for z in range(binary_mask.shape[1]):
        # Extract the 2D slice (z-th plane)
        slice_z = binary_mask[:, z, :]

        # Find contours in the 2D slice
        contours, _ = cv2.findContours(slice_z, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a new blank image to draw the contours
        contour_image = np.zeros((binary_mask.shape[0],binary_mask.shape[2]))
        
        # Draw the contours on the new image
        cv2.drawContours(contour_image, contours, -1, 1, thickness=1)
        # show_one_slice(contour_image,subject)

        # Append the contours to the 3D contours list
        contours_image_list.append(contour_image)
            
    contours_image_final_dim2 = np.array(contours_image_list)
    contours_image_final_dim2 = contours_image_final_dim2.transpose(1,0,2)
    # show_mask_dim1(contours_image_final_dim2,'hej')
    
    
    
    
    #####################################
    ############### DIM 3 ###############
    #####################################
    #Initialise list
    contours_image_list = []
    
    for z in range(binary_mask.shape[2]):
        # Extract the 2D slice (z-th plane)
        slice_z = binary_mask[:, :, z]

        # Find contours in the 2D slice
        contours, _ = cv2.findContours(slice_z, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a new blank image to draw the contours
        contour_image = np.zeros((binary_mask.shape[0],binary_mask.shape[1]))
        
        # Draw the contours on the new image
        cv2.drawContours(contour_image, contours, -1, 1, thickness=1)
        # show_one_slice(contour_image,subject)

        # Append the contours to the 3D contours list
        contours_image_list.append(contour_image)
            
    contours_image_final_dim3 = np.array(contours_image_list)
    contours_image_final_dim3 = contours_image_final_dim3.transpose(1,2,0)
    # show_mask_dim1(contours_image_final_dim3,'hej')
    
    contours_final_list = [contours_image_final_dim1, contours_image_final_dim2, contours_image_final_dim3]
    contour_combined_image = reduce(np.logical_or,contours_final_list)
    # show_mask_dim1(contour_combined_image,'hej')

    return contour_combined_image



def hausdorff_distance_3d(array1, array2):
    """
    Denne funktion kan beregne hausdorff distance!

    Parameters
    ----------
    array1: 3D numpy array
        Den ene af de to arrays som KONTURER! Dvs. kør ovenstående funktion find_contours først!
        
    array2: 3D numpy array
        Den anden af de to arrays som KONTURER! Dvs. kør ovenstående funktion find_contours først!
        
    Returns
    -------
    hausdorff_distance : float
        Hausdorff-distancen!
    
    """
    
    # Convert 3D numpy arrays to 3D point clouds
    surface_X = np.transpose(np.nonzero(array1))
    surface_Y = np.transpose(np.nonzero(array2))
    
    # Calculate pairwise distances between points
    distancesX_to_Y = cdist(surface_X, surface_Y)
    distancesY_to_X = cdist(surface_Y, surface_X)
    
    # Find the maximum distances in both directions
    hausdorff_X_to_Y = np.max(np.min(distancesX_to_Y, axis=1))
    hausdorff_Y_to_X = np.max(np.min(distancesY_to_X, axis=1))
    
    # Hausdorff distance is the maximum of both directions
    hausdorff_distance = max(hausdorff_X_to_Y, hausdorff_Y_to_X)
    
    return hausdorff_distance
     



['L6','L5','L4','L3','L2','L1','T12','T11','T10','T9','T8','T7','T6'],

mapping_Verse = {
'sub-gl279': ['L6','L5','L4','L2','L1','T12','T11','T10','T9'],
'sub-verse502': ['NONE','L5','L4','L3','L2','L1','T12','T11','T10'],
'sub-verse509': ['L6','L5','L4','L3','L2','L1','T12','T11','T10','T9','T8','T7','T6'],
'sub-verse512': ['L5','L4','L3','L2','L1','T12','T11','T10','T9','T8','T7'],
'sub-verse517': ['L6','L5','L4','L3','L2','L1','T12','T11','T10','T9','T8','T7','T6'],
'sub-verse526': ['L5','L4','L3','L2','L1','T12','T11','T10','T9','T8','T7'],
'sub-verse540': ['L6','L5','L4','L3','L2','L1','T12','T11','T10','T9','T8','T7'],
'sub-verse551': ['L6','L5','L4','L3','L2','L1','T12','T11','T10','T9','T8','T7','T6'],
'sub-verse555': ['NONE','L5','L4','L3','L2','L1','T12','T11','T10','T9'],
'sub-verse558': ['L5','L4','L3','L2','L1','T12','T11','T10','T9','T8','T7'],
'sub-verse560': ['L6','L5','L4','L3','L2','L1','T12','T11','T10','T9','T8','T7','T6'],
'sub-verse570': ['L6','L5','L4','L3','L2','L1','T12','T11','T10','T9','T8','T7','T6','T5'],
'sub-verse578': ['L6','L5','L4','L3','L2','L1','T12','T11','T10'],
'sub-verse582': ['L5','L4','L3','L2','L1','T13','T12','T11','T10','T9','T8','T7'], #HAR T13!
'sub-verse587': ['L5','L4','L3','L2','L1','T12','T11','T10','T9'],
'sub-verse590': ['NONE','L5','L4','L3','L2','L1','T12','T11','T10'],
'sub-verse599': ['NONE','L5','L4','L3','L2','L1','T12','T11','T10','T9','T8','T7'],
'sub-verse604': ['L6','L5','L4','L3','L2','L1','T12','T11','T10','NONE'],
'sub-verse607': ['L6','L5','L4','L3','L2','L1','T12','T11','T10','T9','T8','T7'],
'sub-verse613': ['L6','L5','L4','L3','L2','L1','T12','T11','T10','T9','T8','T7','T6'],
'sub-verse616': ['NONE','L5','L4','L3','L2','L1','T12','T11','T10'],
'sub-verse620': ['L6','L5','L4','L3','L2','L1','T12','T11','T10','T9','T8'],
'sub-verse626': ['L5','L4','L3','L2','L1','T11','T10','T9','T8','T7'], #MANGLER T12!
'sub-verse649': ['L5','L4','L3','L2','L1','T12','T11','T10','T9','T8','T7','T6'],
'sub-verse702': ['L5','L4','L3','L2','L1','T12','T11','T10','NONE'],
'sub-verse704': ['L5','L4','L3','L2','L1','T12','T11','T10','T9'],
'sub-verse708': ['L5','L4','L3','L2','L1','T12','T11','T10'],
'sub-verse712': ['L5','L4','L3','L2','L1','T12','T11','T10','T9'],
'sub-verse714': ['L5','L4','L3','L2','L1','T12','T11','T10','T9','NONE'],
'sub-verse716': ['L5','L4','L3','L2','L1','T12','T11','T10'],
'sub-verse752': ['L5','L4','L3','L2','L1','T12','T11','T10','T9','T8','T7','T6','T5'],
'sub-verse756': ['L5','L4','L3','L2','L1','T12','T11','T10','T9'],
'sub-verse760': ['L5','L4','L3','L2','L1','T12','T11','T10'],
'sub-verse762': ['L5','L4','L3','L2','L1','T12','T11','T10'],
'sub-verse766': ['L5','L4','L3','L2','L1','T12','T11','T10','T9','T8','T7','T6','T5'],
'sub-verse768': ['L6','L5','L4','L3','L2','L1','T12','T11','T10','T9','T8','NONE'],
'sub-verse804': ['L5','L4','L3','L2','L1','T12','T11','T10','T9','T8','T7','T6'],
'sub-verse809': ['L5','L4','L3','L2','L1','T12','T11','T10','T9','T8','T7','T6','T5'],
'sub-verse810': ['L6','L5','L4','L3','L2','L1','T12','T11','T10'],
'sub-verse813': ['L5','L4','L3','L2','L1','T12','T11','T10','T9'],
}


['L6','L5','L4','L3','L2','L1','T12','T11','T10','T9','T8','T7','T6'],
['L5','L4','L3','L2','L1','T12'],
['L5','L4','L3','L2','L1','T12','T11'],
['L5','L4','L3','L2','L1','T12','T11','T10'],

#NONE hvis partly visible kun
mapping_RH = {
"VERTEBRAE_LOWHU_0100_SERIES0018": ['L5','L4','L3','L2','L1','T12','T11'],
"VERTEBRAE_LOWHU_0101_SERIES0012": ['L5','L4','L3','L2','L1','T12'],
"VERTEBRAE_LOWHU_0102_SERIES0019": ['L5','L4','L3','L2','L1','T12'],
"VERTEBRAE_LOWHU_0103_SERIES0011": ['L5','L4','L3','L2','L1','NONE'],
"VERTEBRAE_LOWHU_0105_SERIES0013": ['L5','L4','L3','L2','L1','T12','T11'],
"VERTEBRAE_LOWHU_0106_SERIES0021": ['L5','L4','L3','L2','L1','T12'],
"VERTEBRAE_LOWHU_0107_SERIES0022": ['L5','L4','L3','L2','L1','NONE'],
"VERTEBRAE_LOWHU_0108_SERIES0010": ['L5','L4','L3','L2','L1','NONE'],
"VERTEBRAE_LOWHU_0109_SERIES0023": ['L5','L4','L3','L2','L1','T12','T11','NONE'], #T10 er lidt partly. Men kun lidt
"VERTEBRAE_LOWHU_0110_SERIES0017": ['L5','L4','L3','L2','L1','T12'],
"VERTEBRAE_LOWHU_0111_SERIES0016": ['L5','L4','L3','L2','L1','T12','NONE'],
"VERTEBRAE_LOWHU_0112_SERIES0013": ['L5','L4','L3','L2','L1','T12'],
"VERTEBRAE_LOWHU_0113_SERIES0021": ['L5','L4','L3','L2','L1','T12','T11','NONE'],
"VERTEBRAE_LOWHU_0114_SERIES0019": ['L5','L4','L3','L2','L1','T12'],
"VERTEBRAE_LOWHU_0115_SERIES0017": ['L5','L4','L3','L2','L1','T12'],
"VERTEBRAE_LOWHU_0116_SERIES0022": ['L5','L4','L3','L2','L1','NONE'], #Meget lidt partly
"VERTEBRAE_LOWHU_0117_SERIES0021": ['L5','L4','L3','L2','L1','T12','NONE'],
"VERTEBRAE_LOWHU_0118_SERIES0026": ['L5','L4','L3','L2','L1','T12','NONE'],
"VERTEBRAE_LOWHU_0119_SERIES0014": ['L5','L4','L3','L2','L1','T12','T11','T10'], #MEGET I TVIVL!!
"VERTEBRAE_LOWHU_0120_SERIES0022": ['L5','L4','L3','L2','L1','T12'], #Meget lidt partly
"VERTEBRAE_LOWHU_0121_SERIES0023": ['L5','L4','L3','L2','L1','T12','NONE'],
"VERTEBRAE_LOWHU_0122_SERIES0015": ['NONE','L5','L4','L3','L2','L1','T12','T11'],
"VERTEBRAE_LOWHU_0123_SERIES0026": ['L5','L4','L3','L2','L1','T12'],
"VERTEBRAE_LOWHU_0124_SERIES0020": ['L5','L4','L3','L2','L1','T12','T11'],
"VERTEBRAE_LOWHU_0126_SERIES0011": ['L5','L4','L3','L2','L1','T12'],
"VERTEBRAE_LOWHU_0127_SERIES0022": ['L5','L4','L3','L2','L1','T12','NONE'],
"VERTEBRAE_LOWHU_0128_SERIES0015": ['L5','L4','L3','L2','L1','T12'],
"VERTEBRAE_LOWHU_0130_SERIES0021": ['L5','L4','L3','L2','L1','T12'],
"VERTEBRAE_LOWHU_0131_SERIES0017": ['L5','L4','L3','L2','L1','T12','NONE'],
"VERTEBRAE_LOWHU_0133_SERIES0013": ['L5','L4','L3','L2','L1','T12','NONE'],
"VERTEBRAE_LOWHU_0134_SERIES0010": ['L5','L4','L3','L2','L1','T12','T11'],
"VERTEBRAE_LOWHU_0136_SERIES0025": ['L5','L4','L3','L2','L1','NONE'],
"VERTEBRAE_LOWHU_0137_SERIES0010": ['L5','L4','L3','L2','L1','T12','T11'],
"VERTEBRAE_LOWHU_0138_SERIES0026": ['L5','L4','L3','L2','L1'], #Går godt men meget metal her
"VERTEBRAE_LOWHU_0139_SERIES0016": ['L5','L4','L3','L2','L1','T12','T11','T10'], #Tror den er hel. Ikke helt sikker
"VERTEBRAE_LOWHU_0140_SERIES0023": ['L5','L4','L3','L2','L1','T12'], #Tror den er hel. Ikke helt sikker
"VERTEBRAE_LOWHU_0141_SERIES0016": ['L5','L4','L3','L2','L1','T12','NONE'],
"VERTEBRAE_LOWHU_0142_SERIES0023": ['L5','L4','L3','L2','L1','T12','T11'],
"VERTEBRAE_LOWHU_0143_SERIES0011": ['L5','L4','L3','L2','L1','T12','NONE'],
"VERTEBRAE_LOWHU_0144_SERIES0010": ['L5','L4','L3','L2','L1','T12','T11'],
"VERTEBRAE_LOWHU_0146_SERIES0010": ['L5','L4','L3','L2','L1','T12'],
"VERTEBRAE_LOWHU_0147_SERIES0013": ['L4','L3','L2','L1','T12'], #Går fint men meget metal
"VERTEBRAE_LOWHU_0149_SERIES0015": ['NONE','L5','L4','L3','L2','L1','T12','NONE'],
"VERTEBRAE_LOWHU_0151_SERIES0010": ['L5','L4','L3','L2','L1','NONE'], #Meget lidt partly
"VERTEBRAE_LOWHU_0153_SERIES0016": ['L5','L4','L3','L2','L1','NONE'],
"VERTEBRAE_LOWHU_0154_SERIES0015": ['L5','L4','L3','L2','L1','T12','NONE'],
"VERTEBRAE_LOWHU_0155_SERIES0016": ['L5','L4','L3','L2','L1','T12'],
"VERTEBRAE_LOWHU_0156_SERIES0025": ['L5','L4','L3','L2','L1','T12','NONE'],
"VERTEBRAE_LOWHU_0157_SERIES0008": ['L5','L4','L3','L2','L1','T12'],
"VERTEBRAE_LOWHU_0158_SERIES0014": ['L5','L4','L3','L2','L1','T12'],
"VERTEBRAE_LOWHU_0159_SERIES0006": ['L5','L4','L3','L2','L1','T12','T11'],
"VERTEBRAE_LOWHU_0160_SERIES0015": ['L5','L4','L3','L2','L1','T12'],
"VERTEBRAE_LOWHU_0161_SERIES0020": ['L5','L4','L3','L2','L1','NONE'], #Meget lidt partly
"VERTEBRAE_LOWHU_0162_SERIES0018": ['L5','L4','L3','L2','L1','T12','T11'],
"VERTEBRAE_LOWHU_0163_SERIES0020": ['L5','L4','L3','L2','L1','T12','NONE'],
"VERTEBRAE_LOWHU_0164_SERIES0002": ['L5','L4','L3','L2','L1','T12'],
"VERTEBRAE_LOWHU_0165_SERIES0020": ['L5','L4','L3','L2','L1','T12','NONE'],
"VERTEBRAE_LOWHU_0166_SERIES0011": ['L5','L4','L3','L2','L1','T12','NONE'],
"VERTEBRAE_LOWHU_0167_SERIES0013": ['L5','L4','L3','L2','L1'],
"VERTEBRAE_LOWHU_0168_SERIES0014": ['L5','L4','L3','L2','L1','T12'],
"VERTEBRAE_LOWHU_0169_SERIES0007": ['L5','L4','L3','L2','L1','T12'],
"VERTEBRAE_LOWHU_0170_SERIES0006": ['L5','L4','L3','L2','L1','T12','NONE'],
"VERTEBRAE_LOWHU_0171_SERIES0011": ['L5','L4','L3','L2','L1','T12'],
"VERTEBRAE_LOWHU_0172_SERIES0008": ['L5','L4','L3','L2','L1','T12','T11','NONE'],
"VERTEBRAE_LOWHU_0173_SERIES0019": ['L5','L4','L3','L2','L1','T12','NONE'],
"VERTEBRAE_LOWHU_0174_SERIES0008": ['L5','L4','L3','L2','L1','T12'],
"VERTEBRAE_LOWHU_0175_SERIES0016": ['L5','L4','L3','L2','L1','T12','NONE'],
"VERTEBRAE_LOWHU_0176_SERIES0025": ['L5','L4','L3','L2','L1','T12','T11','NONE'],
"VERTEBRAE_LOWHU_0177_SERIES0019": ['L5','L4','L3','L2','L1','T12'],
"VERTEBRAE_LOWHU_0178_SERIES0010": ['L5','L4','L3','L2','L1','T12','NONE'],
"VERTEBRAE_LOWHU_0179_SERIES0019": ['L5','L4','L3','L2','L1','T12'],
"VERTEBRAE_LOWHU_0180_SERIES0014": ['NONE','L5','L4','L3','L2','L1','T12'],
"VERTEBRAE_LOWHU_0181_SERIES0009": ['L5','L4','L3','L2','L1','T12'], #Meget hvid knogle på billedet
"VERTEBRAE_LOWHU_0182_SERIES0010": ['L5','L4','L3','L2','L1','T12','T11'],
"VERTEBRAE_LOWHU_0183_SERIES0011": ['L5','L4','L3','L2','L1','T12','T11'],
"VERTEBRAE_LOWHU_0184_SERIES0019": ['L5','L4','L3','L2','L1','NONE'],
"VERTEBRAE_LOWHU_0185_SERIES0018": ['L5','L4','L3','L2','L1','T12'],
"VERTEBRAE_LOWHU_0186_SERIES0009": ['L5','L4','L3','L2','L1'],
"VERTEBRAE_LOWHU_0187_SERIES0022": ['L5','L4','L3','L2','L1','NONE'],
"VERTEBRAE_LOWHU_0188_SERIES0017": ['L5','L4','L3','L2','L1','T12','NONE'],
"VERTEBRAE_LOWHU_0189_SERIES0025": ['L5','L4','L3','L2','L1','T12'], #KAN IKKE SE BUNDEN!
"VERTEBRAE_LOWHU_0190_SERIES0020": ['L5','L4','L3','L2','L1','T12'],
"VERTEBRAE_LOWHU_0191_SERIES0003": ['L1','T12'], #Meget metal
"VERTEBRAE_LOWHU_0192_SERIES0010": ['L5','L4','L3','L2','L1','NONE'],
"VERTEBRAE_LOWHU_0193_SERIES0023": ['L5','L4','L3','L2','L1','T12'],
"VERTEBRAE_LOWHU_0194_SERIES0012": ['L5','L4','L3','L2','L1','NONE'],
"VERTEBRAE_LOWHU_0195_SERIES0022": ['L5','L4','L3','L2','L1','NONE'],
"VERTEBRAE_LOWHU_0197_SERIES0019": ['L5','L4','L3','L2','L1','T12'],
"VERTEBRAE_LOWHU_0198_SERIES0021": ['L5','L4','L3','L2','L1','NONE'],
"VERTEBRAE_LOWHU_0199_SERIES0008": ['L5','L4','L3','L2','L1','T12','NONE'],
"VERTEBRAE_FRACTURE_0206_SERIES0005": ['L5','L4','L3','L2','L1','T12','NONE'],
"VERTEBRAE_FRACTURE_0207_SERIES0013": ['L5','L4','L3','L2','L1','T12'], #Næsten ikke synlig
"VERTEBRAE_FRACTURE_0208_SERIES0007": ['L5','L4','L3','L2','L1','T12','T11','T10'],
"VERTEBRAE_FRACTURE_0209_SERIES0011": ['L5','L4','L3','L2','L1','T12','T11'],
"VERTEBRAE_FRACTURE_0210_SERIES0012": ['L5','L4','L3','L2','L1','T12'],
"VERTEBRAE_FRACTURE_0211_SERIES0009": ['L5','L4','L3','L2','L1','T12','T11'],
"VERTEBRAE_FRACTURE_0212_SERIES0012": ['L5','L4','L3','L2','L1','T12','T11'],
"VERTEBRAE_FRACTURE_0213_SERIES0008": ['L5','L4','L3','L2','L1','T12','T11'],
"VERTEBRAE_FRACTURE_0214_SERIES0012": ['L5','L4','L3','L2','L1','NONE'], #L1 kun lidt synlig
"VERTEBRAE_FRACTURE_0215_SERIES0005": ['L5','L4','L3','L2','L1','T12','NONE'],
"VERTEBRAE_FRACTURE_0216_SERIES0000": ['L5','L4','L3','L2','L1','T12'],
"VERTEBRAE_FRACTURE_0217_SERIES0001": ['L5','L4','L3','L2','L1','T12','NONE'],
"VERTEBRAE_FRACTURE_0218_SERIES0013": ['L5','L4','L3','L2','L1','T12','NONE'],
"VERTEBRAE_FRACTURE_0219_SERIES0007": ['L5','L4','L3','L2','L1','NONE'],
"VERTEBRAE_FRACTURE_0224_SERIES0000": ['L5','L4','L3','L2','L1','T12','T11'], #I TVIVL OM DEN HER! FJERN!
"VERTEBRAE_FRACTURE_0225_SERIES0014": ['L5','L4','L3','L2','L1','T12','NONE'], #Kun halvt synlig
"VERTEBRAE_FRACTURE_0226_SERIES0013": ['L5','L4','L3','L2','L1','T12','T11','NONE'],
"VERTEBRAE_FRACTURE_0227_SERIES0010": ['L5','L4','L3','L2','L1','T12','NONE'],
"VERTEBRAE_FRACTURE_0228_SERIES0014": ['L5','L4','L3','L2','L1','T12','T11','NONE'],
"VERTEBRAE_FRACTURE_0229_SERIES0007": ['L5','L4','L3','L2','L1','T12','T11'],
"VERTEBRAE_FRACTURE_0236_SERIES0013": ['L5','L4','L3','L2','L1','T12','NONE'],
"VERTEBRAE_FRACTURE_0237_SERIES0005": ['L5','L4','L3','L2','L1','T12','T11','NONE'],
"VERTEBRAE_FRACTURE_0239_SERIES0003": ['L5','L4','L3','L2','L1','T12','T11'],
"VERTEBRAE_FRACTURE_0240_SERIES0010": ['L5','L4','L3','L2','L1','T12','NONE'],
"VERTEBRAE_FRACTURE_0241_SERIES0004": ['L5','L4','L3','L2','L1','T12','T11'],
"VERTEBRAE_FRACTURE_0242_SERIES0020": ['L5','L4','L3','L2','L1','T12','T11', 'NONE'], #Kun halvt synlig
"VERTEBRAE_FRACTURE_0243_SERIES0023": ['L5','L4','L3','L2','L1','T12','NONE'], #Kun halvt synlig
"VERTEBRAE_FRACTURE_0244_SERIES0015": ['L5','L4','L3','L2','L1','T12','T11'],
"VERTEBRAE_FRACTURE_0245_SERIES0013": ['L5','L4','L3','L2','L1','T12','NONE'],
"VERTEBRAE_FRACTURE_0246_SERIES0015": ['L5','L4','L3','L2','L1','T12','NONE'],
"VERTEBRAE_FRACTURE_0247_SERIES0021": ['L5','L4','L3','L2','L1','T12','NONE'],
"VERTEBRAE_FRACTURE_0248_SERIES0008": ['L5','L4','L3','L2','L1','T12','T11','NONE'],
"VERTEBRAE_FRACTURE_0249_SERIES0013": ['L5','L4','L3','L2','L1','NONE'], #FINDER IKKE KÆMPE FRACTURE
"VERTEBRAE_FRACTURE_0251_SERIES0017": ['L5','L4','L3','L2','L1','T12','T11'],
"VERTEBRAE_FRACTURE_0252_SERIES0016": ['L5','L4','L3','L2','L1','NONE'],
"VERTEBRAE_FRACTURE_0253_SERIES0018": ['L5','L4','L3','L2','L1','T12','NONE'],
"VERTEBRAE_FRACTURE_0254_SERIES0015": ['L5','L4','L3','L2','L1','T12','T11'],
"VERTEBRAE_FRACTURE_0255_SERIES0012": ['T12','NONE'], #REMOVE
"VERTEBRAE_FRACTURE_0256_SERIES0019": ['L5','L4','L3','L2','L1'],
"VERTEBRAE_FRACTURE_0257_SERIES0017": ['L5','L4','L3','L2','L1','T12','NONE'], #Kun lidt synlig men fin
"VERTEBRAE_FRACTURE_0258_SERIES0022": ['L5','L4','L3','L2','L1','T12'],
"VERTEBRAE_FRACTURE_0261_SERIES0017": ['L4','L3','L2','T12'], 
"VERTEBRAE_FRACTURE_0262_SERIES0000": ['L5','L4','L3','L2','L1','T12','T11'],
"VERTEBRAE_FRACTURE_0263_SERIES0016": ['L5','L4','L3','L2','L1','T12','T11'],
"VERTEBRAE_FRACTURE_0264_SERIES0010": ['L5','L4','L3','L2','L1','T12'],
"VERTEBRAE_FRACTURE_0265_SERIES0010": ['L5','L4','L3','L2','L1','NONE'],
"VERTEBRAE_FRACTURE_0266_SERIES0019": ['L5','L4','L3','L2','L1','T12','T11', 'NONE'],
"VERTEBRAE_FRACTURE_0267_SERIES0021": ['L5','L4','L3','L2','L1','T12','NONE'],
"VERTEBRAE_FRACTURE_0269_SERIES0022": ['L5','L4','L3','L2','L1','T12'],
"VERTEBRAE_FRACTURE_0270_SERIES0013": ['L5','L4','L3','L2','L1','T12'],
"VERTEBRAE_FRACTURE_0271_SERIES0012": ['L5','L4','L3','L2','L1','NONE'],
"VERTEBRAE_FRACTURE_0272_SERIES0015": ['L5','L4','L3','L2','L1','T12','NONE'],
"VERTEBRAE_FRACTURE_0273_SERIES0010": ['L5','L4','L3','L2','L1','T12','NONE'],
"VERTEBRAE_FRACTURE_0274_SERIES0024": ['L5','L4','L3','L2','L1','T12','T11','NONE'],
"VERTEBRAE_FRACTURE_0275_SERIES0008": ['L5','L4','L3','L2','L1','T12'],
"VERTEBRAE_FRACTURE_0276_SERIES0021": ['L4','L3','L2','L1'], #ONE LABEL SHIFT?
"VERTEBRAE_FRACTURE_0277_SERIES0017": ['L5','L4','L3','L2','L1','T12'],
"VERTEBRAE_FRACTURE_0278_SERIES0009": ['L5','L4','L3','L2','L1','NONE'],
"VERTEBRAE_FRACTURE_0279_SERIES0001": ['L5','L4','L3','L2','L1','T12'],
"VERTEBRAE_FRACTURE_0280_SERIES0007": ['L5','L4','L3','L2','L1','NONE'],
"VERTEBRAE_FRACTURE_0281_SERIES0020": ['L5','L4','L3','L2','L1','NONE'],
"VERTEBRAE_FRACTURE_0282_SERIES0024": ['L5','L4','L3','L2','L1','T12','T11','NONE'], #T10 kun lidt synlig. L1 ser også sjov ud
"VERTEBRAE_FRACTURE_0283_SERIES0018": ['L5','L4','L3','L2','L1','T12'], #God fracture i L4 den her
"VERTEBRAE_FRACTURE_0284_SERIES0024": ['L5','L4','L3','L2','L1'], 
"VERTEBRAE_FRACTURE_0285_SERIES0018": ['L5','L4','L3','L2','L1','T12','T11'],
"VERTEBRAE_FRACTURE_0286_SERIES0011": ['L5','L4','L3','L2','L1','NONE'],
"VERTEBRAE_FRACTURE_0287_SERIES0022": ['L5','L4','L3','L2','L1','T12','NONE'],
"VERTEBRAE_FRACTURE_0288_SERIES0014": ['L5','L4','L3','L2','L1','T12','T11'], #Dårlig kvalitet
"VERTEBRAE_FRACTURE_0289_SERIES0017": ['L4','L3','L2','L1','T12'], #Dårlig kvalitet
"VERTEBRAE_FRACTURE_0290_SERIES0016": ['L5','L4','L3','L2','L1','T12','NONE'],
"VERTEBRAE_FRACTURE_0291_SERIES0023": ['L5','L4','L3','L2','L1','T12','T11'],
"VERTEBRAE_FRACTURE_0292_SERIES0011": ['L5','L4','L3','L2','L1','T12'],
"VERTEBRAE_FRACTURE_0293_SERIES0011": ['L5','L4','L3','L2','L1','T12','T11'],
"VERTEBRAE_FRACTURE_0294_SERIES0019": ['L5','L4','L3','L2','L1','T12','T11'],
"VERTEBRAE_FRACTURE_0295_SERIES0021": ['L5','L4','L3','L2','L1','T12','T11', 'NONE'],
"VERTEBRAE_FRACTURE_0296_SERIES0015": ['L5','L4','L3','L2','L1','T12','NONE'], #God
"VERTEBRAE_FRACTURE_0297_SERIES0007": ['L5','L4','L3','L2','L1','T12','T11'],
"VERTEBRAE_FRACTURE_0298_SERIES0019": ['L5','L4','L3','L2','L1','T12','T11', 'NONE'],
"VERTEBRAE_FRACTURE_0299_SERIES0000": ['L5','L4','L3','L2','L1','T12','T11'],
"VERTEBRAE_FRACTURE_0300_SERIES0014": ['L5','L4','L3','L2','L1','T12','NONE'],
"VERTEBRAE_FRACTURE_0301_SERIES0012": ['L5','L4','L3','L2','L1','T12','NONE'],
"VERTEBRAE_FRACTURE_0302_SERIES0011": ['L5','L4','L3','L2','L1','NONE'],
"VERTEBRAE_FRACTURE_0303_SERIES0020": ['L5','L4','L3','L2','L1','T12','T11','T10'],
"VERTEBRAE_FRACTURE_0304_SERIES0021": ['L5','L4','L3','L2','L1','T12','T11'], #Tæt på at være skåret af. Men fin
"VERTEBRAE_FRACTURE_0305_SERIES0021": ['L5','L4','L3','L2','L1','T12','T11'], #Dårlig kvalitet
"VERTEBRAE_FRACTURE_0306_SERIES0019": ['L5','L4','L3','NONE','T12','T11','NONE'], #Kan ikke finde fordi store fractures!!!! T10 tæt på at være helt synlig
"VERTEBRAE_FRACTURE_0307_SERIES0020": ['L5','L4','L3','L2','L1','T12','T11'],
"VERTEBRAE_FRACTURE_0308_SERIES0025": ['L5','L4','L3','L2','L1','T12'],
"VERTEBRAE_FRACTURE_0309_SERIES0021": ['L5','L4','L3','L2','L1','T12'],
"VERTEBRAE_FRACTURE_0310_SERIES0020": ['L5','L4','L3','L2','L1','T12'],
"VERTEBRAE_FRACTURE_0311_SERIES0013": ['L5','L4','L3','L2','L1','T12','T11'], #One label shift. Nej. Men dårlig kvalitet!
"VERTEBRAE_FRACTURE_0312_SERIES0019": ['L5','L4','L3','L2','L1'],
"VERTEBRAE_FRACTURE_0313_SERIES0021": ['L5','L4','L3','L2','L1','NONE'], #Tæt på at være helt synlig
"VERTEBRAE_FRACTURE_0314_SERIES0017": ['L5','L4','L3','L2','L1','T12'],
"VERTEBRAE_FRACTURE_0315_SERIES0012": ['L5','L4','L3','L2','L1','T12','NONE'],
"VERTEBRAE_FRACTURE_0316_SERIES0013": ['L5','L4','L3','L2','L1','T12','T11'], #Super dårlig kvalitet
"VERTEBRAE_FRACTURE_0317_SERIES0022": ['L5','L4','L3','L2','L1','NONE'], #Rigtig god!
"VERTEBRAE_FRACTURE_0318_SERIES0017": ['L5','L4','L3','L2','L1','T12'],
"VERTEBRAE_FRACTURE_0319_SERIES0022": ['L5','L4'], #Går lidt galt i detekteringen
"VERTEBRAE_FRACTURE_0320_SERIES0009": ['L5','L4','L3','L2','L1','T12','T11','NONE'],
"VERTEBRAE_FRACTURE_0321_SERIES0018": ['L5','L4','L3','L2','L1','T12','NONE'], #God. T11 er næsten synlig
"VERTEBRAE_FRACTURE_0322_SERIES0019": ['L5','L4','L3','L2','L1','NONE'], #Ikke nødvendigvis fracture. Måske de der mærkelige huller? I dont know
"VERTEBRAE_FRACTURE_0323_SERIES0018": ['L5','L4','L3','L2','L1','T12','NONE'],
"VERTEBRAE_FRACTURE_0324_SERIES0029": ['L5','L4','L3','L2','L1','T12'],
"VERTEBRAE_FRACTURE_0325_SERIES0013": ['L5','L4','L3','L2','L1'],
"VERTEBRAE_FRACTURE_0326_SERIES0017": ['NONE'], #FJERN. DOES NOT WORK. Dårlig prediction.
"VERTEBRAE_FRACTURE_0327_SERIES0011": ['L5','L4'], #Går lidt galt. Er ellers en flot nok scanning.
"VERTEBRAE_FRACTURE_0328_SERIES0010": ['L5','L4','L3','L2','L1','T12','T11'],
"VERTEBRAE_FRACTURE_0329_SERIES0013": ['L3','L2','L1','NONE'], #Går lidt galt. Fanger ikke de nederste to. T12 er tæt på at være synlig.
"VERTEBRAE_FRACTURE_0330_SERIES0013": ['L5','L4','L3','L2','L1','T12','T11'],
"VERTEBRAE_FRACTURE_0331_SERIES0013": ['L5','L4','L3','L2','L1','T12'],
"VERTEBRAE_FRACTURE_0332_SERIES0006": ['L5','L4','L3','L2','L1','T12'], #Super tydelig aorta. Vær opmærksom!
"VERTEBRAE_FRACTURE_0333_SERIES0018": ['L5','L4','L3','L2','L1','T12','T11','NONE'],
"VERTEBRAE_FRACTURE_0334_SERIES0010": ['L4','L3','L2','L1','T12','T11', 'T10'], #Kunne godt ligne en L6, den ikke finder. Vær opmærksom!!!
"VERTEBRAE_FRACTURE_0335_SERIES0017": ['L5','L4','L3','L2','L1','T12'],
"VERTEBRAE_FRACTURE_0336_SERIES0019": ['L5','L4','L3','L2','L1','NONE'], #T12 er faktisk skåret lidt over...
"VERTEBRAE_FRACTURE_0337_SERIES0016": ['L5','L4','L3','L2','L1','NONE'],
}






# mapping_batchnorm = {
# 'sub-gl279': ['L6','L5','L4','L2','L1','T12','T11','T10'], #Rigtigt at L3 mangler. Den er sammenvokset med L2
# 'sub-verse502': ['L5','L4','L3','L2','L1','T12','T11','T10'],
# 'sub-verse509': ['L6','L5','L4','L3','L2','L1','T12','T11','T10','T9','T8','T7'],
# 'sub-verse512': ['L5','L4','L3','L2','L1','T12','T11','T10','T9'],
# 'sub-verse517': ['L6','L5','L4','L3','L2','L1','T12','T11','T10'],
# 'sub-verse526': ['L5','L4','L3','L2','L1','T12','T11','T10','T9','T8'],
# 'sub-verse540': ['L6','L5','L4','L3','L2','L1','T12','T11','T10','T9'],
# 'sub-verse551': ['L6','L5','L4','L3','L2','L1','T12','T11','T10','T9'],
# 'sub-verse555': ['NONE','L5','NONE','NONE','L3','L2','L1','T12','T11','T10','T9'],
# 'sub-verse558': ['L5','L4','L3','L2','L1','T12','T11','T10','T9'],
# 'sub-verse560': ['L6','L5','L4','L3','L2','L1','T12','T11','T10','T9'],
# 'sub-verse570': ['L6','L5','L4','L3','L2','L1','T12','T11','T10','T9','T8','T7'],
# 'sub-verse578': ['L6','L5','L4','L3','L2','L1','T12','T11','T10'],
# 'sub-verse582': ['L5','L4','L3','L2','L1','T13','T12','T11','T10','T9'], #HAR T13!
# 'sub-verse587': ['L5','L4','L3','L2','L1','T12','T11','T10','T9'],
# 'sub-verse590': ['NONE','L5','L4','L3','L2','L1','T12','T11','T10'],
# 'sub-verse599': ['NONE','L5','L4','L3','L2','L1','T12','T11','T10','T9'],
# 'sub-verse604': ['L6','L5','L4','L3','L2','L1','T12','T11','T10'],
# 'sub-verse607': ['L6','L5','L4','L3','L2','L1','T12','T11','T10','T9','T8'],
# 'sub-verse613': ['L6','L5','L4','L3','L2','L1','T12','T11','T10','T9','T8'],
# 'sub-verse616': ['L5','L4','L3','L2','L1','T12','T11','T10'],
# 'sub-verse620': ['L6','L5','L4','L3','L2','L1','T12','T11','T10','T9'],
# 'sub-verse626': ['L5','L4','L3','L2','L1','T11','T10','T9','T8'], #MANGLER T12!
# 'sub-verse649': ['NONE','L5','L4','L3','L2','L1','T12','T11','T10','T9','T8','T7'],
# 'sub-verse702': ['L5','L4','L3','L2','L1','T12','T11','T10'],
# 'sub-verse704': ['L5','L4','L3','L2','L1','T12','T11','T10','T9'],
# 'sub-verse708': ['L5','L4','L3','L2','L1','T12','T11','T10'],
# 'sub-verse712': ['L5','L4','L3','L2','L1','T12','T11','T10','T9'],
# 'sub-verse714': ['L5','L4','L3','L2','L1','T12','T11','T10','T9'],
# 'sub-verse716': ['L5','L4','L3','L2','L1','T12','T11','T10','NONE'], #NONE, MEN BURDE VÆRE T9.. MEN TAGER DEN VÆK UANSET HVAD.. Er det her ikke forkert? Eller var den decideret ikke med? Det var den nok ikke. Med i elastic!
# 'sub-verse752': ['L5','L4','L3','L2','L1','T12','T11','T10','T9','T8'],
# 'sub-verse756': ['L5','L4','L3','L2','L1','T12','T11','T10','T9'],
# 'sub-verse760': ['L5','L4','L3','L2','L1','T12','T11','T10'],
# 'sub-verse762': ['L5','L4','L3','L2','L1','T12','T11','T10'],
# 'sub-verse766': ['L5','L4','L3','L2','L1','T12','T11','T10','T9','T8','T7','T6'],
# 'sub-verse768': ['L6','L5','L4','L3','L2','L1','T12','T11','T10','T9'],
# 'sub-verse804': ['L5','L4','L3','L2','L1','T12','T11','T10','T9','T8'],
# 'sub-verse809': ['L5','L4','L3','L2','L1','T12','T11','T10','T9','T8','T7'],
# 'sub-verse810': ['L6','L5','L4','L3','L2','L1','T12','T11','T10'],
# 'sub-verse813': ['L5','L4','L3','L2','L1','T12','T11','T10','T9']
# }


# mapping_elastic = {
# 'sub-gl279': ['L6','L5','L4','L2','L1','T12','T11','T10'],
# 'sub-verse502': ['L5','L4','L3','L2','L1','T12','T11','T10'],
# 'sub-verse509': ['L6','L5','L4','L3','L2','L1','T12','T11','T10','T9','T8','T7','T6'],
# 'sub-verse512': ['L5','L4','L3','L2','L1','T12','T11','T10','T9','T8','T7'],
# 'sub-verse517': ['L6','L5','L4','L3','L2','L1','T12','T11','T10','T9','T8','T7','T6'],
# 'sub-verse526': ['L5','L4','L3','L2','L1','T12','T11','T10','T9','T8','T7'],
# 'sub-verse540': ['L6','L5','L4','L3','L2','L1','T12','T11','T10','T9','T8','T7'],
# 'sub-verse551': ['L6','L5','L4','L3','L2','L1','T12','T11','T10','T9','T8','T7','T6'],
# 'sub-verse555': ['NONE','L5','L4','L3','L2','L1','T12','T11','T10','T9'], #Måske skal jeg tage L4 med? Den her er ret off.. Tag det med i discussion!!!
# 'sub-verse558': ['L5','L4','L3','L2','L1','T12','T11','T10','T9','T8','T7'],
# 'sub-verse560': ['L6','L5','L4','L3','L2','L1','T12','T11','T10','T9','T8','T7','T6'],
# 'sub-verse570': ['L6','L5','L4','L3','L2','L1','T12','T11','T10','T9','T8','T7','T6'],
# 'sub-verse578': ['L6','L5','L4','L3','L2','L1','T12','T11','T10'],
# 'sub-verse582': ['L5','L4','L3','L2','L1','T13','T12','T11','T10','T9','T8','T7'], #OBS T13
# 'sub-verse587': ['L5','L4','L3','L2','L1','T12','T11','T10','T9'],
# 'sub-verse590': ['NONE','L5','L4','L3','L2','L1','T12','T11','T10'],
# 'sub-verse599': ['NONE','L5','L4','L3','L2','L1','T12','T11','T10','T9','T8','T7'],
# 'sub-verse604': ['L6','L5','L4','L3','L2','L1','T12','T11','T10','NONE'],
# 'sub-verse607': ['L6','L5','L4','L3','L2','L1','T12','T11','T10','T9','T8'],
# 'sub-verse613': ['L6','L5','L4','L3','L2','L1','T12','T11','T10','T9','T8','T7','T6'],
# 'sub-verse616': ['L5','L4','L3','L2','L1','T12','T11','T10'],
# 'sub-verse620': ['L6','L5','L4','L3','L2','L1','T12','T11','T10','T9'],
# 'sub-verse626': ['L5','L4','L3','L2','L1','T11','T10','T9','T8','T7'], #Mangler T12
# 'sub-verse649': ['L5','L4','L3','L2','L1','T12','T11','T10','T9','T8','T7'],
# 'sub-verse702': ['L5','L4','L3','L2','L1','T12','T11','T10'],
# 'sub-verse704': ['L5','L4','L3','L2','L1','T12','T11','T10','T9'],
# 'sub-verse708': ['L5','L4','L3','L2','L1','T12','T11','T10'],
# 'sub-verse712': ['L5','L4','L3','L2','L1','T12','T11','T10','T9'],
# 'sub-verse714': ['L5','L4','L3','L2','L1','T12','T11','T10','T9','NONE'],
# 'sub-verse716': ['L5','L4','L3','L2','L1','T12','T11','T10'],
# 'sub-verse752': ['L5','L4','L3','L2','L1','T12','T11','T10','T9','T8','T7','T6'],
# 'sub-verse756': ['L5','L4','L3','L2','L1','T12','T11','T10','T9'],
# 'sub-verse760': ['L5','L4','L3','L2','L1','T12','T11','T10'],
# 'sub-verse762': ['L5','L4','L3','L2','L1','T12','T11','T10'],
# 'sub-verse766': ['L5','L4','L3','L2','L1','T12','T11','T10','T9','T8','T7','T6','T5'],
# 'sub-verse768': ['L6','L5','L4','L3','L2','L1','T12','T11','T10','T9','T8'],
# 'sub-verse804': ['L5','L4','L3','L2','L1','T12','T11','T10','T9','T8'],
# 'sub-verse809': ['L5','L4','L3','L2','L1','T12','T11','T10','T9','T8','T7','T6'],
# 'sub-verse810': ['L6','L5','L4','L3','L2','L1','T12','T11','T10'],
# 'sub-verse813': ['L5','L4','L3','L2','L1','T12','T11','T10']
# }