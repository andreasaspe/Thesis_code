import os
import nibabel as nib
import nibabel.orientations as nio
from data_utilities import *
# from nibabel.affines import apply_affine
# import numpy.linalg as npl
import pickle
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from my_plotting_functions import *
#import SimpleITK as sitk



#######################################################
#################### CONTROL PANEL ####################
#######################################################
#Define scans
all_scans = 0 #Set to 1 if you want to preprocess all scans
list_of_subjects = ['sub-verse507'] #List of subjects 521

#Define rescale and reorientation parameters
New_orientation = ('L', 'A', 'S')
New_voxel_size = 8 # [mm]

#If you want to print output
Print_info = 0 #Set to 1 to print a lot of info on each scan.

#Define preprocessing details and printing output[s
# OBS, set everthing to 1. Also remember to set all_scans to 1 if you want to preprocess all scans.
Preprocess = 1 #Set to 1 if you want to do actual preprocessing and save it. Set to 0 for just plotting.
normalize_HU = 1 #Set to 1 if you want to normalize to the below range
HU_range_normalize = [-1, 1]
HU_cutoff = 1 #Set to 1 if you want to cut_off_HU. Define below:
HU_range_cutoff = [-200, 1000] #Define HU range. Only takes effekt if HU_cutoff is set to 1
pad_value = -200 #-1024 #'minimum' #Put in number or the string 'minimum' for padding with the minimum value in volume
dim1_new = 64
dim2_new = 64
dim3_new = 128


#Define directories
#Cluster
# dir_data = '/scratch/s174197/data/Verse20/Verse20_training_unpacked' #'/scratch/s174197/data/Verse20/Verse20_training_unpacked' #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_test_unpacked' #'/scratch/s174197/data/Verse20/Verse20_test_unpacked' #'/scratch/s174197/data/Verse20/Verse20_test_unpacked' #'/Users/andreasaspe/Documents/Data/Verse20_training' #r'C:\Users\PC\Documents\Andreas_s174197\dataset-verse20training_unpacked' #'/zhome/bb/f/127616/Documents/Thesis/Rawdata_training'
# Output_folder = '/scratch/s174197/data/Verse20/SpineLocalisation/Verse20_training_prep' #'/scratch/s174197/data/Verse20/Verse20_test_prep' #'/Users/andreasaspe/Documents/Data/Verse20_training_prep' #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data' #"/Users/andreasaspe/Documents/Data/Preprocessed_data"
# Padding_output_directory = '/scratch/s174197/data/Verse20/SpineLocalisation/Padding_specifications'
# Padding_output_filename = 'pad_training'
#mac
dir_data = '/Users/andreasaspe/Documents/Data/Verse20/Verse20_training_unpacked'
Output_folder = '/scratch/s174197/data/Verse20/SpineLocalisation/Verse20_training_prep' #'/scratch/s174197/data/Verse20/Verse20_test_prep' #'/Users/andreasaspe/Documents/Data/Verse20_training_prep' #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data' #"/Users/andreasaspe/Documents/Data/Preprocessed_data"


#Below is not for preprocessing.
save_reoriented = 0
Reoriented_folder = '/scratch/s174197/data/Verse20/Verse20_test_reoriented'
save_rescaled = 0
Rescaled_folder = '/scratch/s174197/data/Verse20/Verse20_test_rescaled'
save_reoriented_and_rescaled = 0
Reoriented_and_rescaled_folder = '/scratch/s174197/data/Verse20/Verse20_test_reoriented_and_rescaled'

##### PLOTTING ######
Slices = 40 #Slices per volume
plot_mask = 0 #Set to 1 for plotting mask
plot_dim1_before = 0
plot_dim2_before = 0
plot_dim3_before = 0
plot_dim1_after = 1
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
        subject = filename.split("_")[0]
        #if subject.find('verse') != -1:
        all_subjects.append(subject)
    all_subjects = np.unique(all_subjects)
    all_subjects = all_subjects[all_subjects != '.DS'] #Sorting out .DS
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
for subject in all_subjects:
    print("\n\n")
    print("       SUBJECT: "+str(subject)+"\n")
    # LOAD FILES
    # Define file names
    filename_img = [f for f in listdir(dir_data) if (f.startswith(subject) and f.endswith('img.nii.gz'))][0]
    filename_msk = [f for f in listdir(dir_data) if (f.startswith(subject) and f.endswith('msk.nii.gz'))][0]
    filename_ctd = [f for f in listdir(dir_data) if (f.startswith(subject) and f.endswith('json'))][0]
    # Load files
    img_nib = nib.load(os.path.join(dir_data,filename_img))
    msk_nib = nib.load(os.path.join(dir_data,filename_msk))
    ctd_list = load_centroids(os.path.join(os.path.join(dir_data,filename_ctd)))
    
    #Hounsfield units
    # img_data = np.asanyarray(img_nib.dataobj, dtype=img_nib.dataobj.dtype)
    # img_data[img_data<0] = 0
    # img_data[img_data>1000] = 1000
    # img_nib = nib.Nifti1Image(img_data, img_nib.affine)


    #Get info
    zooms = img_nib.header.get_zooms() #Voxel sizes
    axs_code = nio.ornt2axcodes(nio.io_orientation(img_nib.affine)) #Image orientation
    ctd_code = ctd_list[0] #Centroid orientation
    data_shape_voxels = img_nib.header.get_data_shape() #Shape of data
    data_shape_mm = np.array(data_shape_voxels)*np.array(zooms) #Data measures
    data_type = img_nib.header.get_data_dtype() #Data type
    
    #Gaussian smoothing
    # img_data = np.asanyarray(img_nib.dataobj, dtype=img_nib.dataobj.dtype) #Load 
    # img_data = gaussian_filter(img_data, sigma=[0.75/zooms[0] , 0.75/zooms[1], 0.75/zooms[2]]) #3/8 er bedre
    # img_nib = nib.Nifti1Image(img_data, img_nib.affine)
    
    
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
        print('centroids orientation: {}'.format(ctd_code))
        print('img data shape in voxels: {}'.format(data_shape_voxels))
        print('img data shape in mm: {}'.format(data_shape_mm))
        print('img data type: {}'.format(data_type))
        print("\n")
                                                                                                                                                                                                                    
    #Load data for plotting
    if plot_dim1_before or plot_dim2_before or plot_dim3_before:
        img_data = img_nib.get_fdata() #Load data
        msk_data = msk_nib.get_fdata() #Load data
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
            if plot_mask == 1:
                ax.imshow(msk_data[i,:,:].T,cmap="jet",origin="lower",alpha =0.5*(msk_data[i,:,:].T>0))
            ax.set_title('Dim1, '+str(subject)+", Slice: "+str(i))
            for v in ctd_list[1:]:
                ax.add_patch(Circle((v[2],v[3]), 7*1/zooms[0]))
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
            if plot_mask == 1:
                ax.imshow(msk_data[:,i,:].T,cmap="jet",origin="lower",alpha =0.5*(msk_data[:,i,:].T>0))
            ax.set_title('Dim2, '+str(subject)+", Slice: "+str(i))
            for v in ctd_list[1:]:
                ax.add_patch(Circle((v[1],v[3]), 7*1/zooms[1]))
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
            if plot_mask == 1:
                ax.imshow(msk_data[:,:,i].T,cmap="jet",origin="lower",alpha =0.5*(msk_data[:,:,i].T>0))
            ax.set_title('Dim3, '+str(subject)+", Slice: "+str(i))
            for v in ctd_list[1:]:
                ax.add_patch(Circle((v[1],v[2]), 7*1/zooms[2]))
            plt.show()


    # # #Do stuff before resampling:
    # data_img = np.asanyarray(img_nib.dataobj, dtype=img_nib.dataobj.dtype)
    # #Gaussian smoothing
    # sigma = [0.75/zooms[0],0.75/zooms[1],0.75/zooms[2]]
    # data_img_gauss = gaussian_filter(data_img, sigma=sigma)
    # #Change hounsfield units
    # data_img[data_img<HU_range_cutoff[0]] = HU_range_cutoff[0]
    # data_img[data_img>HU_range_cutoff[1]] = HU_range_cutoff[1]
    # # #Gaussian smoothing
    # data_img = gaussian_filter(data_img, sigma=2) #3/8 er bedre  0.75/8
    
    
    # img_nib = nib.Nifti1Image(data_img, img_nib.affine)
    


    #RESAMPLED
    if rescale:
        vs = (New_voxel_size,New_voxel_size,New_voxel_size)
        print("Rescaling image")
        img_resampled = resample_nib(img_nib, voxel_spacing=vs, order=3)
        msk_resampled = resample_nib(msk_nib, voxel_spacing=vs, order=0) # or resample based on img: resample_mask_to(msk_nib, img_iso)
        ctd_resampled = rescale_centroids(ctd_list, img_nib, vs) #Hmmm
        
        #Get info
        zooms = img_resampled.header.get_zooms() #Voxel sizes
        axs_code = nio.ornt2axcodes(nio.io_orientation(img_resampled.affine)) #Image orientation
        ctd_code = ctd_resampled[0] #Centroid orientation
        data_shape_voxels = img_resampled.header.get_data_shape() #Shape of data
        data_shape_mm = np.array(data_shape_voxels)*np.array(zooms) #Data measures
        data_type = img_resampled.header.get_data_dtype() #Data type
            
        #Print info
        if Print_info:
            print("After rescaling:")
            print('img orientation: {}'.format(axs_code))
            print('centroids orientation: {}'.format(ctd_code))
            print('img data shape in voxels: {}'.format(data_shape_voxels))
            print('img data shape in mm: {}'.format(data_shape_mm))
            print('img data type: {}'.format(data_type))
            print("\n")

        #Get data and find dimensions
        if plot_dim1_rescaled or plot_dim2_rescaled or plot_dim3_rescaled or save_rescaled:
            img_data_resampled = np.asanyarray(img_resampled.dataobj, dtype=img_resampled.dataobj.dtype)
            msk_resampled = np.asanyarray(msk_resampled.dataobj, dtype=msk_resampled.dataobj.dtype)
            dim1, dim2, dim3 = img_data_resampled.shape
        
        if save_rescaled:
            img_nifti = nib.Nifti1Image(img_data_resampled, img_data_resampled.affine)
            msk_nifti = nib.Nifti1Image(img_data_resampled, img_data_resampled.affine)
            
            #Define folders
            img_path = os.path.join(Rescaled_folder,'img')
            msk_path = os.path.join(Rescaled_folder,'msk')
            ctd_path = os.path.join(Rescaled_folder,'ctd')
            #Create output-folders if it does not exist
            if not os.path.exists(img_path):
               os.makedirs(img_path)
            if not os.path.exists(msk_path):
               os.makedirs(msk_path)
            if not os.path.exists(ctd_path):
                os.makedirs(ctd_path)
                
            #Save data
            nib.save(img_nifti, os.path.join(Rescaled_folder, img_path, subject+'reoriented_img.nii.gz'))
            nib.save(msk_nifti, os.path.join(Rescaled_folder, msk_path, subject+'reoriented_msk.nii.gz'))
            save_centroids(ctd_resampled, os.path.join(Rescaled_folder, ctd_path, subject+'reoriented_ctd.json'))
            
        if plot_dim1_rescaled:
            slice_step = int(dim1/Slices)
            if slice_step == 0:
                slice_step = 1
            max_val = img_data_resampled.max()
            min_val = img_data_resampled.min()
            for i in range(0,dim1,slice_step):
                fig, ax = plt.subplots()
                ax.imshow(img_data_resampled[i,:,:].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
                if plot_mask == 1:
                    ax.imshow(msk_resampled[i,:,:].T,cmap="jet",origin="lower",alpha =0.5*(msk_resampled[i,:,:].T>0))
                ax.set_title('Rescaled Dim1, '+str(subject)+", Slice: "+str(i))
                for v in ctd_resampled[1:]:
                    ax.add_patch(Circle((v[2],v[3]), 7*1/zooms[0]))
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
                if plot_mask == 1:
                    ax.imshow(msk_resampled[:,i,:].T,cmap="jet",origin="lower",alpha =0.5*(msk_resampled[:,i,:].T>0))
                ax.set_title('Rescaled Dim2, '+str(subject)+", Slice: "+str(i))
                for v in ctd_resampled[1:]:
                    ax.add_patch(Circle((v[1],v[3]), 7*1/zooms[1]))
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
                if plot_mask == 1:
                    ax.imshow(msk_resampled[:,:,i].T,cmap="jet",origin="lower",alpha =0.5*(msk_resampled[:,:,i].T>0))
                ax.set_title('Rescaled Dim3, '+str(subject)+", Slice: "+str(i))
                for v in ctd_resampled[1:]:
                    ax.add_patch(Circle((v[1],v[2]), 7*1/zooms[2]))
                plt.show()
            
    #REORIENTATION
    if reorient:
        print("Reorienting image")
        img_reoriented = reorient_to(img_nib, axcodes_to=New_orientation)
        msk_reoriented = reorient_to(msk_nib, axcodes_to=New_orientation)
        ctd_reoriented = reorient_centroids_to(ctd_list, img_reoriented)
        
        #Get info
        zooms = img_reoriented.header.get_zooms() #Voxel sizes
        axs_code = nio.ornt2axcodes(nio.io_orientation(img_reoriented.affine)) #Image orientation
        ctd_code = ctd_reoriented[0] #Centroid orientation
        data_shape_voxels = img_reoriented.header.get_data_shape() #Shape of data
        data_shape_mm = np.array(data_shape_voxels)*np.array(zooms) #Data measures
        data_type = img_reoriented.header.get_data_dtype() #Data type
        
        #Print info
        if Print_info:
            print("After reorientation:")
            print('img orientation: {}'.format(axs_code))
            print('centroids orientation: {}'.format(ctd_code))
            print('img data shape in voxels: {}'.format(data_shape_voxels))
            print('img data shape in mm: {}'.format(data_shape_mm))
            print('img data type: {}'.format(data_type))
            print("\n")
            
        #Load data for plotting or saving
        if plot_dim1_reoriented or plot_dim2_reoriented or plot_dim3_reoriented or save_reoriented:
            img_data_reoriented = np.asanyarray(img_reoriented.dataobj, dtype=img_reoriented.dataobj.dtype)
            msk_data_reoriented = np.asanyarray(msk_reoriented.dataobj, dtype=msk_reoriented.dataobj.dtype)
            dim1, dim2, dim3 = img_data_reoriented.shape
    
        if save_reoriented:
            img_nifti = nib.Nifti1Image(img_data_reoriented, img_reoriented.affine)
            msk_nifti = nib.Nifti1Image(msk_data_reoriented, msk_reoriented.affine)
            
            #Define folders
            img_path = os.path.join(Reoriented_folder,'img')
            msk_path = os.path.join(Reoriented_folder,'msk')
            ctd_path = os.path.join(Reoriented_folder,'ctd')
            #Create output-folders if it does not exist
            if not os.path.exists(img_path):
               os.makedirs(img_path)
            if not os.path.exists(msk_path):
               os.makedirs(msk_path)
            if not os.path.exists(ctd_path):
                os.makedirs(ctd_path)
                
            #Save data
            nib.save(img_nifti, os.path.join(Reoriented_folder, img_path, subject+'_img.nii.gz'))
            nib.save(msk_nifti, os.path.join(Reoriented_folder, msk_path, subject+'_msk.nii.gz'))
            save_centroids(ctd_reoriented, os.path.join(Reoriented_folder, ctd_path, subject+'_ctd.json'))
            
        if plot_dim1_reoriented:
            slice_step = int(dim1/Slices)
            if slice_step == 0:
                slice_step = 1
            max_val = img_data_reoriented.max()
            min_val = img_data_reoriented.min()
            for i in range(0,dim1,slice_step):
                fig, ax = plt.subplots()
                ax.imshow(img_data_reoriented[i,:,:].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
                if plot_mask == 1:
                    ax.imshow(msk_data_reoriented[i,:,:].T,cmap="jet",origin="lower",alpha =0.5*(msk_reoriented[i,:,:].T>0))
                ax.set_title('Reorient Dim1, '+str(subject)+", Slice: "+str(i))
                for v in ctd_reoriented[1:]:
                    ax.add_patch(Circle((v[2],v[3]), 7*1/zooms[0]))
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
                if plot_mask == 1:
                    ax.imshow(msk_data_reoriented[:,i,:].T,cmap="jet",origin="lower",alpha =0.5*(msk_reoriented[:,i,:].T>0))
                ax.set_title('Reorient Dim2, '+str(subject)+", Slice: "+str(i))
                for v in ctd_reoriented[1:]:
                    ax.add_patch(Circle((v[1],v[3]), 7*1/zooms[1]))
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
                if plot_mask == 1:
                    ax.imshow(msk_data_reoriented[:,:,i].T,cmap="jet",origin="lower",alpha =0.5*(msk_reoriented[:,:,i].T>0))
                ax.set_title('Reorient Dim3, '+str(subject)+", Slice: "+str(i))
                for v in ctd_reoriented[1:]:
                    ax.add_patch(Circle((v[1],v[2]), 7*1/zooms[2]))
                plt.show()
                
                
                
    if rescale_and_reorient:
        print("Rescaling and reorienting")
        if rescale==0:
            vs = (New_voxel_size,New_voxel_size,New_voxel_size)
            img_resampled = resample_nib(img_nib, voxel_spacing=vs, order=3)
            msk_resampled = resample_nib(msk_nib, voxel_spacing=vs, order=0) # or resample based on img: resample_mask_to(msk_nib, img_iso)
            ctd_resampled = rescale_centroids(ctd_list, img_nib, vs)
            
        img_resampled_reoriented = reorient_to(img_resampled, axcodes_to=New_orientation)
        msk_resampled_reoriented = reorient_to(msk_resampled, axcodes_to=New_orientation)
        ctd_resampled_reoriented = reorient_centroids_to(ctd_resampled, img_resampled_reoriented)
        
        
        #Get info
        zooms = img_resampled_reoriented.header.get_zooms() #Voxel sizes
        axs_code = nio.ornt2axcodes(nio.io_orientation(img_resampled_reoriented.affine)) #Image orientation
        ctd_code = ctd_resampled_reoriented[0] #Centroid orientation
        data_shape_voxels = img_resampled_reoriented.header.get_data_shape() #Shape of data
        data_shape_mm = np.array(data_shape_voxels)*np.array(zooms) #Data measures
        data_type = img_resampled_reoriented.header.get_data_dtype() #Data type
        
        #Print info
        if Print_info:
            print("AFTER RESCALING AND REORIENTATION:")
            print('img orientation: {}'.format(axs_code))
            print('centroids orientation: {}'.format(ctd_code))
            print('img data shape in voxels: {}'.format(data_shape_voxels))
            print('img data shape in mm: {}'.format(data_shape_mm))
            print('img data type: {}'.format(data_type))
            print("\n")
            
        #Load data for plotting or saving
        if plot_dim1_rescaled_and_reoriented or plot_dim2_rescaled_and_reoriented or plot_dim3_rescaled_and_reoriented or save_reoriented_and_rescaled:
            img_data_resampled_reoriented = np.asanyarray(img_resampled_reoriented.dataobj, dtype=img_resampled_reoriented.dataobj.dtype) #Load 
            msk_data_resampled_reoriented = np.asanyarray(msk_resampled_reoriented.dataobj, dtype=msk_resampled_reoriented.dataobj.dtype) #Load 
            dim1, dim2, dim3 = img_data_resampled_reoriented.shape #Find dimensions
        
        if save_reoriented_and_rescaled:
            img_nifti = nib.Nifti1Image(img_data_reoriented, img_resampled_reoriented.affine)
            msk_nifti = nib.Nifti1Image(msk_data_reoriented, msk_resampled_reoriented.affine)
            
            #Define folders
            img_path = os.path.join(Reoriented_and_rescaled_folder,'img')
            msk_path = os.path.join(Reoriented_and_rescaled_folder,'msk')
            ctd_path = os.path.join(Reoriented_and_rescaled_folder,'ctd')
            #Create output-folders if it does not exist
            if not os.path.exists(img_path):
               os.makedirs(img_path)
            if not os.path.exists(msk_path):
               os.makedirs(msk_path)
            if not os.path.exists(ctd_path):
                os.makedirs(ctd_path)
                
            #Save data
            nib.save(img_nifti, os.path.join(Reoriented_and_rescaled_folder, img_path, subject+'_img.nii.gz'))
            nib.save(msk_nifti, os.path.join(Reoriented_and_rescaled_folder, msk_path, subject+'_.nii.gz'))
            save_centroids(ctd_resampled_reoriented, os.path.join(Reoriented_and_rescaled_folder, ctd_path, subject+'_ctd.json'))
    
    
        if plot_dim1_rescaled_and_reoriented:
            slice_step = int(dim1/Slices)
            if slice_step == 0:
                slice_step = 1
            max_val = img_data_resampled_reoriented.max()
            min_val = img_data_resampled_reoriented.min()
            for i in range(0,dim1,slice_step):
                fig, ax = plt.subplots()
                ax.imshow(img_data_resampled_reoriented[i,:,:].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)
                if plot_mask == 1:
                    ax.imshow(msk_data_resampled_reoriented[i,:,:].T,cmap="jet",origin="lower",alpha =0.5*(msk_data_resampled_reoriented[i,:,:].T>0))
                ax.set_title('Rescale + reorient Dim1, '+str(subject)+", Slice: "+str(i))
                # for v in ctd_resampled_reoriented[1:]:
                #     ax.add_patch(Circle((v[2],v[3]), 7*1/zooms[0]))
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
                if plot_mask == 1:
                    ax.imshow(msk_data_resampled_reoriented[:,i,:].T,cmap="jet",origin="lower",alpha =0.5*(msk_data_resampled_reoriented[:,i,:].T>0))
                ax.set_title('Rescale + reorient Dim2, '+str(subject)+", Slice: "+str(i))
                for v in ctd_resampled_reoriented[1:]:
                    ax.add_patch(Circle((v[1],v[3]), 7*1/zooms[2]))
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
                if plot_mask == 1:
                    ax.imshow(msk_data_resampled_reoriented[:,i,:].T,cmap="jet",origin="lower",alpha =0.5*(msk_data_resampled_reoriented[:,i,:].T>0))
                ax.set_title('Rescale + reorient Dim3, '+str(subject)+", Slice: "+str(i))
                for v in ctd_resampled_reoriented[1:]:
                    ax.add_patch(Circle((v[1],v[2]), 7*1/zooms[2]))
                plt.show()
                
        
         #Save model
        if Preprocess:
            # img_path = os.path.join(Output_folder,'img')
            # msk_path = os.path.join(Output_folder,'msk')
            # ctd_path = os.path.join(Output_folder,'ctd')
            # #Create output-folders if it does not exist
            # if not os.path.exists(img_path):
            #     os.makedirs(img_path)
            # if not os.path.exists(msk_path):
            #     os.makedirs(msk_path)
            # if not os.path.exists(ctd_path):
            #     os.makedirs(ctd_path)

            #Loading data
            data_img = np.asanyarray(img_resampled_reoriented.dataobj, dtype=img_resampled_reoriented.dataobj.dtype)
            data_msk = np.asanyarray(msk_resampled_reoriented.dataobj, dtype=msk_resampled_reoriented.dataobj.dtype)         
            
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
            
            #Save padding specifications
            restrictions.update({subject: (pad1[0] , pad1[0]+dim1   ,   pad2[0] , pad2[0]+dim2   ,   pad3[0] , pad3[0]+dim3)})

            #Doing padding
            data_img=np.pad(data_img, (pad1, pad2, pad3), constant_values = pad_value)
            data_msk=np.pad(data_msk, (pad1, pad2, pad3), constant_values = pad_value)

            #Find new dimensions
            dim1, dim2, dim3 = data_img.shape
            
            #Change hounsfield units
            if HU_cutoff == 1:
                data_img[data_img<HU_range_cutoff[0]] = HU_range_cutoff[0]
                data_img[data_img>HU_range_cutoff[1]] = HU_range_cutoff[1]

            # #Gaussian smoothing
            # data_img = gaussian_filter(data_img, sigma=0.75/8) #3/8 er bedre

            #Normalise houndsfield units
            if normalize_HU == 1:
                data_img = (HU_range_normalize[1]-HU_range_normalize[0])*(data_img - data_img.min()) / (data_img.max() - data_img.min()) + HU_range_normalize[0]
             
            #Gaussian smoothing
            # data_img = gaussian_filter(data_img, sigma=0.75/vs[0]) #3/8 er bedre. Nu står der 0.75/8
            
            #Define as new Nifti-files
            img_preprocessed = nib.Nifti1Image(data_img, img_resampled_reoriented.affine)
            msk_preprocessed = nib.Nifti1Image(data_msk, msk_resampled_reoriented.affine)
            
            #Change centroids coordinates to fit the padding
            for i in range(len(ctd_resampled_reoriented)-1):
                ctd_resampled_reoriented[i+1][1] += pad1[0]
                ctd_resampled_reoriented[i+1][2] += pad2[0]
                ctd_resampled_reoriented[i+1][3] += pad3[0]
                            
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
                
                #Mask
                zooms = msk_preprocessed.header.get_zooms() #Voxel sizes
                axs_code = nio.ornt2axcodes(nio.io_orientation(msk_preprocessed.affine)) #Image orientation
                data_shape_voxels = msk_preprocessed.header.get_data_shape() #Shape of data
                data_shape_mm = np.array(data_shape_voxels)*np.array(zooms) #Data measures
                data_type = msk_preprocessed.header.get_data_dtype() #Data type
                print("MASK:")
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
                    if plot_mask == 1:
                        ax.imshow(data_msk[i,:,:].T,cmap="jet",origin="lower",alpha =0.5*(data_msk[i,:,:].T>0))
                    ax.set_title('Final Dim1, '+str(subject)+", Slice: "+str(i))
                    for v in ctd_resampled_reoriented[1:]:
                        ax.add_patch(Circle((v[2],v[3]), 7*1/zooms[0]))
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
                    if plot_mask == 1:
                        ax.imshow(data_msk[:,i,:].T,cmap="jet",origin="lower",alpha =0.5*(data_msk[:,i,:].T>0))
                    ax.set_title('Final Dim2, '+str(subject)+", Slice: "+str(i))
                    for v in ctd_resampled_reoriented[1:]:
                        ax.add_patch(Circle((v[1],v[3]), 7*1/zooms[2]))
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
                    if plot_mask == 1:
                        ax.imshow(data_msk[:,i,:].T,cmap="jet",origin="lower",alpha =0.5*(data_msk[:,i,:].T>0))
                    ax.set_title('Final Dim3, '+str(subject)+", Slice: "+str(i))
                    for v in ctd_resampled_reoriented[1:]:
                        ax.add_patch(Circle((v[1],v[2]), 7*1/zooms[2]))
                    plt.show()
                
            #Save data
            nib.save(img_preprocessed, os.path.join(Output_folder, img_path, subject+'_img.nii.gz'))
            nib.save(msk_preprocessed, os.path.join(Output_folder, msk_path, subject+'_msk.nii.gz'))
            save_centroids(ctd_resampled_reoriented, os.path.join(Output_folder, ctd_path, subject+'_ctd.json'))
       
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