import os
import nibabel as nib
import nibabel.orientations as nio
import sys
#sys.path.append('E:/Andreas_s174197/Thesis/MY_CODE/utils')
from data_utilities import *
# from nibabel.affines import apply_affine
# import numpy.linalg as npl
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import pickle
#import SimpleITK as sitk



#######################################################
#################### CONTROL PANEL ####################
#######################################################
#Define scans
all_scans = 0 #Set to 1 if you want to preprocess all scans
list_of_subjects = ['VERTEBRAE_HEALTHY_0001_SERIES0010'] #List of subjects
with open("E:\Andreas_s174197\Thesis\My_code\Other_scripts\list_of_subjects", "rb") as fp:   # Unpickling
    list_of_subjects = pickle.load(fp)

#Define rescale and reorientation parameters
New_orientation = ('L', 'A', 'S')
New_voxel_size = 8 # [mm]

#If you want to print output
Print_info = 1 #Set to 1 to print a lot of info on each scan.

#Define preprocessing details and printing output[s
# OBS, set everthing to 1. Also remember to set all_scans to 1 if you want to preprocess all scans.
Preprocess = 0 #Set to 1 if you want to do actual preprocessing and save it. Set to 0 for just plotting.
normalize_HU = 1 #Set to 1 if you want to normalize to range 0,1
HU_cutoff = 1 #Set to 1 if you want to cut_off_HU. Define below:
HU_range = [-200, 1000] #Define HU range. Only takes effekt if HU_cutoff is set to 1
pad_value = -1024 #'minimum' #Put in number or the string 'minimum' for padding with the minimum value in volume
dim1_new = 64
dim2_new = 64
dim3_new = 128


#Define directories
dir_data = r'F:\DTU-Vertebra-1\NIFTI' #'/zhome/bb/f/127616/Documents/Thesis/Rawdata_training'
Output_folder = r'E:\Andreas_s174197\data_RH\data_prep_temp' #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data' #"/Users/andreasaspe/Documents/Data/Preprocessed_data"
Padding_output_directory = 'E:/Andreas_s174197/data_RH/Padding_specifications'
Padding_output_filename = 'pad_temp'

#Below is not for preprocessing.
save_reoriented = 0
Reoriented_folder = '/scratch/s174197/data/Verse20/Verse20_test_reoriented'
save_rescaled = 0
Rescaled_folder = '/scratch/s174197/data/Verse20/Verse20_test_rescaled'
save_reoriented_and_rescaled = 0
Reoriented_and_rescaled_folder = '/scratch/s174197/data/Verse20/Verse20_test_reoriented_and_rescaled'

##### PLOTTING ######
Slices = 40 #Slices per volume
plot_dim1_before = 0
plot_dim2_before = 0
plot_dim3_before = 0
plot_dim1_after = 0
plot_dim2_after = 0
plot_dim3_after = 0
#Other choices: Plot only rescaling or reorienting (not final preprocessing)
plot_dim1_rescaled = 0
plot_dim2_rescaled = 0
plot_dim3_rescaled = 0
plot_dim1_reoriented = 0
plot_dim2_reoriented = 0
plot_dim3_reoriented = 0
plot_dim1_rescaled_and_reoriented = 0
plot_dim2_rescaled_and_reoriented = 0
plot_dim3_rescaled_and_reoriented = 0
#######################################################
#######################################################
#######################################################
reorient = 0
rescale = 0
rescale_and_reorient = 0


#Fuck op farve: Slice 130 i verse510. Hvis du kører med at se hver 5 slice, så kan man se det!

#Define list of scans
if all_scans:
    all_subjects = []
    for filename in listdir(dir_data):
        subject = filename.split(".")[0]
        all_subjects.append(subject)
    all_subjects = np.unique(all_subjects)
else:
    all_subjects = list_of_subjects

#Check choices for rescaling and reorienting
if plot_dim1_reoriented or plot_dim2_reoriented or plot_dim3_reoriented or save_reoriented:
    reorient = 1

if plot_dim1_rescaled or plot_dim2_rescaled or plot_dim2_rescaled or save_rescaled:
    rescale = 1

if Preprocess or plot_dim1_rescaled_and_reoriented or plot_dim2_rescaled_and_reoriented or plot_dim3_rescaled_and_reoriented or save_reoriented_and_rescaled:
    rescale_and_reorient = 1

#Initialising list and dictionaries to save data on the way
dim1_list = []
dim2_list = []
dim3_list = []
restrictions = {}

#FOR LOOP START
for counter, subject in enumerate(all_subjects):
    print("\n\n")
    print("       SUBJECT: "+str(subject)+"\n")
    # LOAD FILES
    # Define file names
    img_nib = nib.load(os.path.join(dir_data,subject+'.nii.gz'))

            
    #Get info
    zooms = img_nib.header.get_zooms() #Voxel sizes
    axs_code = nio.ornt2axcodes(nio.io_orientation(img_nib.affine)) #Image orientation
    data_shape_voxels = img_nib.header.get_data_shape() #Shape of data
    data_shape_mm = np.array(data_shape_voxels)*np.array(zooms) #Data measures
    data_type = img_nib.header.get_data_dtype() #Data type

    if axs_code[0] in ['L','R']:
        LR = 0
    if axs_code[0] in ['A','P']:
        AP = 0
    if axs_code[0] in ['S','I']:
        SI = 0

    if axs_code[1] in ['L','R']:
        LR = 1
    if axs_code[1] in ['A','P']:
        AP = 1
    if axs_code[1] in ['S','I']:
        SI = 1

    if axs_code[2] in ['L','R']:
        LR = 2
    if axs_code[2] in ['A','P']:
        AP = 2
    if axs_code[2] in ['S','I']:
        SI = 2
    
    dim1_list_new = data_shape_mm[LR]
    dim2_list_new = data_shape_mm[AP]
    dim3_list_new = data_shape_mm[SI]
    dim1_list.append(dim1_list_new)
    dim2_list.append(dim2_list_new)
    dim3_list.append(dim3_list_new)

    if dim1_list_new > 512:
        print(str(subject) + " is too big in dimension 1. Size is "+ str(dim1_list_new))
        continue
    if dim2_list_new > 512:
        print(str(subject) + " is too big in dimension 2. Size is "+ str(dim2_list_new))
        continue
    if dim3_list_new > 1024:
        print(str(subject) + " is too big in dimension 3. Size is "+ str(dim3_list_new))
        continue
    
    #Print info
    if Print_info:
        print("Before any preprocessing:")
        print('img orientation: {}'.format(axs_code))
        print('img data shape in voxels: {}'.format(data_shape_voxels))
        print('img data shape in mm: {}'.format(data_shape_mm))
        print('img data type: {}'.format(data_type))
        print("\n")
                                                                                                                                                                                                                    
    #Load data for plotting
    if plot_dim1_before or plot_dim2_before or plot_dim3_before:
        img_data = np.asanyarray(img_nib.dataobj, dtype=img_nib.dataobj.dtype)
        dim1, dim2, dim3 = img_data.shape #Find dimensions

    #Plot
    if plot_dim1_before:
        slice_step = int(dim1/Slices)
        if slice_step == 0:
            slice_step = 1
        max_val = img_data.max()
        min_val = img_data.min()
        for i in range(0,dim1,slice_step):
            fig, ax = plt.subplots()
            ax.imshow(img_data[i,:,:].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
            ax.set_title('Dim1, '+str(subject)+", Slice: "+str(i))
            plt.show()
    
    if plot_dim2_before:
        slice_step = int(dim2/Slices)
        if slice_step == 0:
            slice_step = 1
        max_val = img_data.max()
        min_val = img_data.min()
        for i in range(0,dim2,slice_step):
            fig, ax = plt.subplots()
            ax.imshow(img_data[:,i,:].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
            ax.set_title('Dim2, '+str(subject)+", Slice: "+str(i))
            plt.show()
    
    if plot_dim3_before:
        slice_step = int(dim3/Slices)
        if slice_step == 0:
            slice_step = 1
        max_val = img_data.max()
        min_val = img_data.min()
        for i in range(0,dim3,slice_step):
            fig, ax = plt.subplots()
            ax.imshow(img_data[:,:,i].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
            ax.set_title('Dim3, '+str(subject)+", Slice: "+str(i))
            plt.show()



    #RESAMPLED
    if rescale:
        vs = (New_voxel_size,New_voxel_size,New_voxel_size)
        print("Rescaling image")
        img_resampled = resample_nib(img_nib, voxel_spacing=vs, order=3)

        #Get info
        zooms = img_resampled.header.get_zooms() #Voxel sizes
        axs_code = nio.ornt2axcodes(nio.io_orientation(img_resampled.affine)) #Image orientation
        data_shape_voxels = img_resampled.header.get_data_shape() #Shape of data
        data_shape_mm = np.array(data_shape_voxels)*np.array(zooms) #Data measures
        data_type = img_resampled.header.get_data_dtype() #Data type
            
        #Print info
        if Print_info:
            print("After rescaling:")
            print('img orientation: {}'.format(axs_code))
            print('img data shape in voxels: {}'.format(data_shape_voxels))
            print('img data shape in mm: {}'.format(data_shape_mm))
            print('img data type: {}'.format(data_type))
            print("\n")

        #Get data and find dimensions
        if plot_dim1_rescaled or plot_dim2_rescaled or plot_dim3_rescaled or save_rescaled:
            img_data_resampled = np.asanyarray(img_resampled.dataobj, dtype=img_resampled.dataobj.dtype)
            dim1, dim2, dim3 = img_data_resampled.shape
        
        if save_rescaled:
            img_nifti = nib.Nifti1Image(img_data_resampled, img_data_resampled.affine)
            
            #Define folders
            img_path = os.path.join(Rescaled_folder,'img')
            #Create output-folders if it does not exist
            if not os.path.exists(img_path):
               os.makedirs(img_path)
                
            #Save data
            nib.save(img_nifti, os.path.join(Rescaled_folder, img_path, subject+'reoriented_img.nii.gz'))
            
        if plot_dim1_rescaled:
            slice_step = int(dim1/Slices)
            if slice_step == 0:
                slice_step = 1
            max_val = img_data_resampled.max()
            min_val = img_data_resampled.min()
            for i in range(0,dim1,slice_step):
                fig, ax = plt.subplots()
                ax.imshow(img_data_resampled[i,:,:].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
                ax.set_title('Rescaled Dim1, '+str(subject)+", Slice: "+str(i))
                plt.show()
                
        if plot_dim2_rescaled:
            slice_step = int(dim2/Slices)
            if slice_step == 0:
                slice_step = 1
            max_val = img_data_resampled.max()
            min_val = img_data_resampled.min()
            for i in range(0,dim2,slice_step):
                fig, ax = plt.subplots()
                ax.imshow(img_data_resampled[:,i,:].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
                ax.set_title('Rescaled Dim2, '+str(subject)+", Slice: "+str(i))
                plt.show()
        
        if plot_dim3_rescaled:
            slice_step = int(dim3/Slices)
            if slice_step == 0:
                slice_step = 1
            max_val = img_data_resampled.max()
            min_val = img_data_resampled.min()
            for i in range(0,dim3,slice_step):
                fig, ax = plt.subplots()
                ax.imshow(img_data_resampled[:,:,i].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
                ax.set_title('Rescaled Dim3, '+str(subject)+", Slice: "+str(i))
                plt.show()
            
    #REORIENTATION
    if reorient:
        print("Reorienting image")
        img_reoriented = reorient_to(img_nib, axcodes_to=New_orientation)

        #Get info
        zooms = img_reoriented.header.get_zooms() #Voxel sizes
        axs_code = nio.ornt2axcodes(nio.io_orientation(img_reoriented.affine)) #Image orientation
        data_shape_voxels = img_reoriented.header.get_data_shape() #Shape of data
        data_shape_mm = np.array(data_shape_voxels)*np.array(zooms) #Data measures
        data_type = img_reoriented.header.get_data_dtype() #Data type
        
        #Print info
        if Print_info:
            print("After reorientation:")
            print('img orientation: {}'.format(axs_code))
            print('img data shape in voxels: {}'.format(data_shape_voxels))
            print('img data shape in mm: {}'.format(data_shape_mm))
            print('img data type: {}'.format(data_type))
            print("\n")
            
        #Load data for plotting or saving
        if plot_dim1_reoriented or plot_dim2_reoriented or plot_dim3_reoriented or save_reoriented:
            img_data_reoriented = np.asanyarray(img_reoriented.dataobj, dtype=img_reoriented.dataobj.dtype)
            dim1, dim2, dim3 = img_data_reoriented.shape
    
        if save_reoriented:
            img_nifti = nib.Nifti1Image(img_data_reoriented, img_reoriented.affine)
            
            #Define folders
            img_path = os.path.join(Reoriented_folder,'img')
            #Create output-folders if it does not exist
            if not os.path.exists(img_path):
               os.makedirs(img_path)
                
            #Save data
            nib.save(img_nifti, os.path.join(Reoriented_folder, img_path, subject+'_img.nii.gz'))
            
        if plot_dim1_reoriented:
            slice_step = int(dim1/Slices)
            if slice_step == 0:
                slice_step = 1
            max_val = img_data_reoriented.max()
            min_val = img_data_reoriented.min()
            for i in range(0,dim1,slice_step):
                fig, ax = plt.subplots()
                ax.imshow(img_data_reoriented[i,:,:].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
                ax.set_title('Reorient Dim1, '+str(subject)+", Slice: "+str(i))
                plt.show()
                
        if plot_dim2_reoriented:
            slice_step = int(dim2/Slices)
            if slice_step == 0:
                slice_step = 1
            max_val = img_data_reoriented.max()
            min_val = img_data_reoriented.min()
            for i in range(0,dim2,slice_step):
                fig, ax = plt.subplots()
                ax.imshow(img_data_reoriented[:,i,:].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
                ax.set_title('Reorient Dim2, '+str(subject)+", Slice: "+str(i))
                plt.show()
        
        if plot_dim3_reoriented:
            slice_step = int(dim3/Slices)
            if slice_step == 0:
                slice_step = 1
            max_val = img_data_reoriented.max()
            min_val = img_data_reoriented.min()
            for i in range(0,dim3,slice_step):
                fig, ax = plt.subplots()
                ax.imshow(img_data_reoriented[:,:,i].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
                ax.set_title('Reorient Dim3, '+str(subject)+", Slice: "+str(i))
                plt.show()
                
                
                
    if rescale_and_reorient:
        print("Rescaling and reorienting")
        if rescale==0:
            vs = (New_voxel_size,New_voxel_size,New_voxel_size)
            img_resampled = resample_nib(img_nib, voxel_spacing=vs, order=3)

        img_resampled_reoriented = reorient_to(img_resampled, axcodes_to=New_orientation)
        
        #Get info
        zooms = img_resampled_reoriented.header.get_zooms() #Voxel sizes
        axs_code = nio.ornt2axcodes(nio.io_orientation(img_resampled_reoriented.affine)) #Image orientation
        data_shape_voxels = img_resampled_reoriented.header.get_data_shape() #Shape of data
        data_shape_mm = np.array(data_shape_voxels)*np.array(zooms) #Data measures
        data_type = img_resampled_reoriented.header.get_data_dtype() #Data type
        
        #Print info
        if Print_info:
            print("AFTER RESCALING AND REORIENTATION:")
            print('img orientation: {}'.format(axs_code))
            print('img data shape in voxels: {}'.format(data_shape_voxels))
            print('img data shape in mm: {}'.format(data_shape_mm))
            print('img data type: {}'.format(data_type))
            print("\n")
            
        #Load data for plotting or saving
        if plot_dim1_rescaled_and_reoriented or plot_dim2_rescaled_and_reoriented or plot_dim3_rescaled_and_reoriented or save_reoriented_and_rescaled:
            img_data_resampled_reoriented = np.asanyarray(img_resampled_reoriented.dataobj, dtype=img_resampled_reoriented.dataobj.dtype) #Load 
            dim1, dim2, dim3 = img_data_resampled_reoriented.shape #Find dimensions
        
        if save_reoriented_and_rescaled:
            img_nifti = nib.Nifti1Image(img_data_reoriented, img_resampled_reoriented.affine)
            
            #Define folders
            img_path = os.path.join(Reoriented_and_rescaled_folder,'img')
            #Create output-folders if it does not exist
            if not os.path.exists(img_path):
               os.makedirs(img_path)
                
            #Save data
            nib.save(img_nifti, os.path.join(Reoriented_and_rescaled_folder, img_path, subject+'_img.nii.gz'))
    
    
        if plot_dim1_rescaled_and_reoriented:
            slice_step = int(dim1/Slices)
            if slice_step == 0:
                slice_step = 1
            max_val = img_data_resampled_reoriented.max()
            min_val = img_data_resampled_reoriented.min()
            for i in range(0,dim1,slice_step):
                fig, ax = plt.subplots()
                ax.imshow(img_data_resampled_reoriented[i,:,:].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
                ax.set_title('Rescale + reorient Dim1, '+str(subject)+", Slice: "+str(i))
                plt.show()
                
        if plot_dim2_rescaled_and_reoriented:
            slice_step = int(dim2/Slices)
            if slice_step == 0:
                slice_step = 1
            max_val = img_data_resampled_reoriented.max()
            min_val = img_data_resampled_reoriented.min()
            for i in range(0,dim2,slice_step):
                fig, ax = plt.subplots()
                ax.imshow(img_data_resampled_reoriented[:,i,:].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
                ax.set_title('Rescale + reorient Dim2, '+str(subject)+", Slice: "+str(i))
                plt.show()
        
        if plot_dim3_rescaled_and_reoriented:
            slice_step = int(dim3/Slices)
            if slice_step == 0:
                slice_step = 1
            max_val = img_data_resampled_reoriented.max()
            min_val = img_data_resampled_reoriented.min()
            for i in range(0,dim3,slice_step):
                fig, ax = plt.subplots()
                ax.imshow(img_data_resampled_reoriented[:,:,i].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
                ax.set_title('Rescale + reorient Dim3, '+str(subject)+", Slice: "+str(i))
                plt.show()
                
        
         #Save model
        if Preprocess:
            img_path = os.path.join(Output_folder,'img')
            #Create output-folders if it does not exist
            if not os.path.exists(img_path):
               os.makedirs(img_path)

            #Loading data
            data_img = np.asanyarray(img_resampled_reoriented.dataobj, dtype=img_resampled_reoriented.dataobj.dtype)
            
            #Find dimensions
            dim1, dim2, dim3 = data_img.shape
            
            #Calculate padding in each side (volume should be centered)
            padding_dim1 = (dim1_new-dim1)/2
            padding_dim2 = (dim2_new-dim2)/2
            padding_dim3 = (dim3_new-dim3)/2
            
            #Calculate padding in each side by taking decimal values into account
            #Dim1
            if np.floor(padding_dim1) == padding_dim1:
                pad1 = (int(padding_dim1),int(padding_dim1))
            else:
                pad1 = (int(np.floor(padding_dim1)),int(np.floor(padding_dim1)+1))
            #Dim2
            if np.floor(padding_dim2) == padding_dim2:
                pad2 = (int(padding_dim2),int(padding_dim2))
            else:
                pad2 = (int(np.floor(padding_dim2)),int(np.floor(padding_dim2)+1))
            #Dim3
            if np.floor(padding_dim3) == padding_dim3:
                pad3 = (int(padding_dim3),int(padding_dim3))
            else:
                pad3 = (int(np.floor(padding_dim3)),int(np.floor(padding_dim3)+1))
            
            #Save padding specificaations
            restrictions.update({subject: (pad1[0] , pad1[0]+dim1   ,   pad2[0] , pad2[0]+dim2   ,   pad3[0] , pad3[0]+dim3)})

            #Doing padding
            data_img=np.pad(data_img, (pad1, pad2, pad3), constant_values = pad_value)

            #Find new dimensions
            dim1, dim2, dim3 = data_img.shape
            
            #Change hounsfield units
            if HU_cutoff == 1:
                data_img[data_img<HU_range[0]] = HU_range[0]
                data_img[data_img>HU_range[1]] = HU_range[1]

            #Normalise houndsfield units
            if normalize_HU == 1:
                data_img = (data_img - data_img.min()) / (data_img.max() - data_img.min())
             
            #Gaussian smoothing
            #data_img = gaussian_filter(data_img, sigma=0.75/8) #3/8 er bedre
            
            #Define as new Nifti-files
            img_preprocessed = nib.Nifti1Image(data_img, img_resampled_reoriented.affine)
                            
            if Print_info:
                #Image
                zooms = img_preprocessed.header.get_zooms() #Voxel sizes
                axs_code = nio.ornt2axcodes(nio.io_orientation(img_preprocessed.affine)) #Image orientation
                data_shape_voxels = img_preprocessed.header.get_data_shape() #Shape of data
                data_shape_mm = np.array(data_shape_voxels)*np.array(zooms) #Data measures
                data_type = img_preprocessed.header.get_data_dtype() #Data type
                
                print("IMAGE:")
                print('img orientation code: {}'.format(axs_code))
                print('img data shape in voxels: {}'.format(data_shape_voxels))
                print('img data shape in mm: {}'.format(data_shape_mm))
                print('img data type: {}'.format(data_type))
                print("\n")

                
            if plot_dim1_after:
                slice_step = int(dim1/Slices)
                if slice_step == 0:
                    slice_step = 1
                max_val = data_img.max()
                min_val = data_img.min()
                for i in range(0,dim1,slice_step):
                    fig, ax = plt.subplots()
                    ax.imshow(data_img[i,:,:].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
                    ax.set_title('Final Dim1, '+str(subject)+", Slice: "+str(i))
                    plt.show()
                    
            if plot_dim2_after:
                slice_step = int(dim2/Slices)
                if slice_step == 0:
                    slice_step = 1
                max_val = data_img.max()
                min_val = data_img.min()
                for i in range(0,dim2,slice_step):
                    fig, ax = plt.subplots()
                    ax.imshow(data_img[:,i,:].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
                    ax.set_title('Final Dim2, '+str(subject)+", Slice: "+str(i))
                    plt.show()
            
            if plot_dim3_after:
                slice_step = int(dim3/Slices)
                if slice_step == 0:
                    slice_step = 1
                max_val = data_img.max()
                min_val = data_img.min()
                for i in range(0,dim3,slice_step):
                    fig, ax = plt.subplots()
                    ax.imshow(data_img[:,:,i].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
                    ax.set_title('Final Dim3, '+str(subject)+", Slice: "+str(i))
                    plt.show()
                
            #Save data
            nib.save(img_preprocessed, os.path.join(Output_folder, img_path, subject+'_img.nii.gz'))

            if counter == 10:
                break
       
# Create the directory if it does not exist
if not os.path.exists(Padding_output_directory):
    os.makedirs(Padding_output_directory)

with open(os.path.join(Padding_output_directory,Padding_output_filename), 'wb') as f:
    pickle.dump(restrictions, f)

            
print("Done")
#Max
print("Dim1 max :"+str(np.max(dim1_list)))
print("Dim2 max :"+str(np.max(dim2_list)))
print("Dim3 max :"+str(np.max(dim3_list)))
#Min
print("Dim1 min :"+str(np.min(dim1_list)))
print("Dim2 min :"+str(np.min(dim2_list)))
print("Dim3 min :"+str(np.min(dim3_list)))
#Mean
print("Dim1 mean :"+str(np.mean(dim1_list)))
print("Dim2 mean :"+str(np.mean(dim2_list)))
print("Dim3 mean :"+str(np.mean(dim3_list)))