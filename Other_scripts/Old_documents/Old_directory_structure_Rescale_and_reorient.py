import os
import nibabel as nib
import nibabel.orientations as nio
from data_utilities import *
# from nibabel.affines import apply_affine
# import numpy.linalg as npl
from os import listdir
import numpy as np
import matplotlib.pyplot as plt




#######################################################
#################### CONTROL PANEL ####################
#######################################################
#Define scans
all_scans = 1 #Set to 1 if you want to preprocess all scans
list_of_subjects = ['sub-verse510'] #List of subjects

#Define rescale and reorientation parameters
New_orientation = ('R', 'A', 'S')
New_voxel_size = 8 # [mm]

#Define preprocessing details and printing outputs
Preprocess = 1 #Set to 1 if you want to do actual preprocessing and save it. Set to 0 for just plotting.
Print_info = 0 #Set to 1 to print a lot of info on each scan.
pad_value = 0
dim1_new = 64
dim2_new = 64
dim3_new = 128

#Define directories
dir_rawdata = '/zhome/bb/f/127616/Downloads/dataset-01training/rawdata/' #'/Users/andreasaspe/Documents/Data/dataset-01training/rawdata'
dir_derivatives = '/zhome/bb/f/127616/Downloads/dataset-01training/derivatives/' #'/Users/andreasaspe/Documents/Data/dataset-01training/derivatives'
Output_folder = '/zhome/bb/f/127616/Documents/Thesis_outputs/' #"/Users/andreasaspe/Documents/Data/Preprocessed_data"


##### PLOTTING ######
Slices = 20 #Slices per volume
plot_mask = 0 #Set to 1 for plotting mask
plot_dim1_before = 0
plot_dim2_before = 0
plot_dim3_before = 0
plot_dim1_after = 0
plot_dim2_after = 0
plot_dim3_after = 0
#Other choices: Plot only rescaling or reorienting (not final preprocessing)
plot_dim1_rescale = 0
plot_dim2_rescale = 0
plot_dim3_rescale = 0
plot_dim1_reoriented = 0
plot_dim2_reoriented = 0
plot_dim3_reoriented = 0
plot_dim1_rescaled_and_reoriented = 0
plot_dim2_rescaled_and_reoriented = 0
plot_dim3_rescaled_and_reoriented = 0
#######################################################
#######################################################
#######################################################

#Fuck op farve: Slice 130 i verse510. Hvis du kører med at se hver 5 slice, så kan man se det!


#Define list of scans
if all_scans:
    scans = [f for f in listdir(dir_rawdata) if f.startswith('sub')] #Remove file .DS_Store
else:
    scans = list_of_subjects


#Check choices for rescaling and reorienting
reorient = 0
rescale = 0
rescale_and_reorient = 0
if plot_dim1_reoriented or plot_dim2_reoriented or plot_dim3_reoriented:
    reorient = 1

if plot_dim1_rescale or plot_dim2_rescale or plot_dim2_rescale:
    rescale = 1

if Preprocess or plot_dim1_rescaled_and_reoriented or plot_dim2_rescaled_and_reoriented or plot_dim3_rescaled_and_reoriented:
    rescale_and_reorient = 1



#FOR LOOP START
for subject in scans:
    print("       SUBJECT: "+str(subject)+"\n")
    # LOAD FILES
    # Define file names
    filename_img = [f for f in listdir(os.path.join(dir_rawdata,subject)) if f.endswith('.gz')][0]
    filename_msk = [f for f in listdir(os.path.join(dir_derivatives,subject)) if f.endswith('.gz')][0]
    filename_ctd = [f for f in listdir(os.path.join(dir_derivatives,subject)) if f.endswith('.json')][0]
    # Load files
    img_nib = nib.load(os.path.join(dir_rawdata,subject,filename_img))
    msk_nib = nib.load(os.path.join(dir_derivatives,subject,filename_msk))
    ctd_list = load_centroids(os.path.join(os.path.join(dir_derivatives,subject,filename_ctd)))
    
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
    
    #Print info
    if Print_info:
        print("Before any preprocessing:")
        print('img orientation code: {}'.format(axs_code))
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
        for i in range(0,dim1,slice_step):
            fig, ax = plt.subplots()
            ax.imshow(img_data[i,:,:].T,cmap="gray",origin="lower")
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
        for i in range(0,dim2,slice_step):
            fig, ax = plt.subplots()
            ax.imshow(img_data[:,i,:].T,cmap="gray",origin="lower")
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
        for i in range(0,dim3,slice_step):
            fig, ax = plt.subplots()
            ax.imshow(img_data[:,:,i].T,cmap="gray",origin="lower")
            if plot_mask == 1:
                ax.imshow(msk_data[:,:,i].T,cmap="jet",origin="lower",alpha =0.5*(msk_data[:,:,i].T>0))
            ax.set_title('Dim3, '+str(subject)+", Slice: "+str(i))
            for v in ctd_list[1:]:
                ax.add_patch(Circle((v[1],v[2]), 7*1/zooms[2]))
            plt.show()


    #RESAMPLED
    if rescale:
        vs = (New_voxel_size,New_voxel_size,New_voxel_size)
        print("Rescaling image")
        img_resampled = resample_nib(img_nib, voxel_spacing=vs, order=3)
        msk_resampled = resample_nib(msk_nib, voxel_spacing=vs, order=0) # or resample based on img: resample_mask_to(msk_nib, img_iso)
        ctd_resampled = rescale_centroids(ctd_list, img_resampled, vs)
        
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
            print('img orientation code: {}'.format(axs_code))
            print('centroids orientation: {}'.format(ctd_code))
            print('img data shape in voxels: {}'.format(data_shape_voxels))
            print('img data shape in mm: {}'.format(data_shape_mm))
            print('img data type: {}'.format(data_type))
            print("\n")

        #Get data and find dimensions
        img_data_resampled = img_resampled.get_fdata()
        msk_resampled = msk_resampled.get_fdata()
        dim1, dim2, dim3 = img_data_resampled.shape
            
        if plot_dim1_rescale:
            slice_step = int(dim1/Slices)
            if slice_step == 0:
                slice_step = 1
            for i in range(0,dim1,slice_step):
                fig, ax = plt.subplots()
                ax.imshow(img_data_resampled[i,:,:].T,cmap="gray",origin="lower")
                if plot_mask == 1:
                    ax.imshow(msk_resampled[i,:,:].T,cmap="jet",origin="lower",alpha =0.5*(msk_resampled[i,:,:].T>0))
                ax.set_title('Rescaled Dim1, '+str(subject)+", Slice: "+str(i))
                for v in ctd_resampled[1:]:
                    ax.add_patch(Circle((v[2],v[3]), 7*1/zooms[0]))
                plt.show()
                
        if plot_dim2_rescale:
            slice_step = int(dim2/Slices)
            if slice_step == 0:
                slice_step = 1
            for i in range(0,dim2,slice_step):
                fig, ax = plt.subplots()
                ax.imshow(img_data_resampled[:,i,:].T,cmap="gray",origin="lower")
                if plot_mask == 1:
                    ax.imshow(msk_resampled[:,i,:].T,cmap="jet",origin="lower",alpha =0.5*(msk_resampled[:,i,:].T>0))
                ax.set_title('Rescaled Dim2, '+str(subject)+", Slice: "+str(i))
                for v in ctd_resampled[1:]:
                    ax.add_patch(Circle((v[1],v[3]), 7*1/zooms[1]))
                plt.show()
        
        if plot_dim3_rescale:
            slice_step = int(dim3/Slices)
            if slice_step == 0:
                slice_step = 1
            for i in range(0,dim3,slice_step):
                fig, ax = plt.subplots()
                ax.imshow(img_data_resampled[:,:,i].T,cmap="gray",origin="lower")
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
            
        #Get data and find dimensions
        img_data_reoriented = img_reoriented.get_fdata()
        msk_reoriented = msk_reoriented.get_fdata()
        dim1, dim2, dim3 = img_data_reoriented.shape
    
        if plot_dim1_reoriented:
            slice_step = int(dim1/Slices)
            if slice_step == 0:
                slice_step = 1
            for i in range(0,dim1,slice_step):
                fig, ax = plt.subplots()
                ax.imshow(img_data_reoriented[i,:,:].T,cmap="gray",origin="lower")
                if plot_mask == 1:
                    ax.imshow(msk_reoriented[i,:,:].T,cmap="jet",origin="lower",alpha =0.5*(msk_reoriented[i,:,:].T>0))
                ax.set_title('Reorient Dim1, '+str(subject)+", Slice: "+str(i))
                for v in ctd_reoriented[1:]:
                    ax.add_patch(Circle((v[2],v[3]), 7*1/zooms[0]))
                plt.show()
                
        if plot_dim2_reoriented:
            slice_step = int(dim2/Slices)
            if slice_step == 0:
                slice_step = 1
            for i in range(0,dim2,slice_step):
                fig, ax = plt.subplots()
                ax.imshow(img_data_reoriented[:,i,:].T,cmap="gray",origin="lower")
                if plot_mask == 1:
                    ax.imshow(msk_reoriented[:,i,:].T,cmap="jet",origin="lower",alpha =0.5*(msk_reoriented[:,i,:].T>0))
                ax.set_title('Reorient Dim2, '+str(subject)+", Slice: "+str(i))
                for v in ctd_reoriented[1:]:
                    ax.add_patch(Circle((v[1],v[3]), 7*1/zooms[1]))
                plt.show()
        
        if plot_dim3_reoriented:
            slice_step = int(dim3/Slices)
            if slice_step == 0:
                slice_step = 1
            for i in range(0,dim3,slice_step):
                fig, ax = plt.subplots()
                ax.imshow(img_data_reoriented[:,:,i].T,cmap="gray",origin="lower")
                if plot_mask == 1:
                    ax.imshow(msk_reoriented[:,:,i].T,cmap="jet",origin="lower",alpha =0.5*(msk_reoriented[:,:,i].T>0))
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
            ctd_resampled = rescale_centroids(ctd_list, img_resampled, vs)
            
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
            print('img orientation code: {}'.format(axs_code))
            print('centroids orientation: {}'.format(ctd_code))
            print('img data shape in voxels: {}'.format(data_shape_voxels))
            print('img data shape in mm: {}'.format(data_shape_mm))
            print('img data type: {}'.format(data_type))
            print("\n")
            
        #Load data for plotting
        if plot_dim1_rescaled_and_reoriented or plot_dim2_rescaled_and_reoriented or plot_dim3_rescaled_and_reoriented:
            img_data_resampled_reoriented = img_resampled_reoriented.get_fdata() #Load 
            msk_resampled_reoriented = msk_resampled_reoriented.get_fdata() #Load 
            dim1, dim2, dim3 = img_data_resampled_reoriented.shape #Find dimensions

        if plot_dim1_rescaled_and_reoriented:
            slice_step = int(dim1/Slices)
            if slice_step == 0:
                slice_step = 1
            for i in range(0,dim1,slice_step):
                fig, ax = plt.subplots()
                ax.imshow(img_data_resampled_reoriented[i,:,:].T,cmap="gray",origin="lower")
                if plot_mask == 1:
                    ax.imshow(msk_resampled_reoriented[i,:,:].T,cmap="jet",origin="lower",alpha =0.5*(msk_resampled_reoriented[i,:,:].T>0))
                ax.set_title('Rescale + reorient Dim1, '+str(subject)+", Slice: "+str(i))
                for v in ctd_resampled_reoriented[1:]:
                    ax.add_patch(Circle((v[2],v[3]), 7*1/zooms[0]))
                plt.show()
                
        if plot_dim2_rescaled_and_reoriented:
            slice_step = int(dim2/Slices)
            if slice_step == 0:
                slice_step = 1
            for i in range(0,dim2,slice_step):
                fig, ax = plt.subplots()
                ax.imshow(img_data_resampled_reoriented[:,i,:].T,cmap="gray",origin="lower")
                if plot_mask == 1:
                    ax.imshow(msk_resampled_reoriented[:,i,:].T,cmap="jet",origin="lower",alpha =0.5*(msk_resampled_reoriented[:,i,:].T>0))
                ax.set_title('Rescale + reorient Dim2, '+str(subject)+", Slice: "+str(i))
                for v in ctd_resampled_reoriented[1:]:
                    ax.add_patch(Circle((v[1],v[3]), 7*1/zooms[2]))
                plt.show()
        
        if plot_dim3_rescaled_and_reoriented:
            slice_step = int(dim3/Slices)
            if slice_step == 0:
                slice_step = 1
            for i in range(0,dim3,slice_step):
                fig, ax = plt.subplots()
                ax.imshow(img_data_resampled_reoriented[:,:,i].T,cmap="gray",origin="lower")
                if plot_mask == 1:
                    ax.imshow(msk_resampled_reoriented[:,i,:].T,cmap="jet",origin="lower",alpha =0.5*(msk_resampled_reoriented[:,i,:].T>0))
                ax.set_title('Rescale + reorient Dim3, '+str(subject)+", Slice: "+str(i))
                for v in ctd_resampled_reoriented[1:]:
                    ax.add_patch(Circle((v[1],v[2]), 7*1/zooms[2]))
                plt.show()
                
        
         #Save model
        if Preprocess:
            #Loading data
            data_img = np.asanyarray(img_resampled_reoriented.dataobj, dtype=img_resampled_reoriented.dataobj.dtype)
            data_msk = np.asanyarray(msk_resampled_reoriented.dataobj, dtype=msk_resampled_reoriented.dataobj.dtype)         
            
            #Find dimensions
            dim1, dim2, dim3 = data_img.shape
            
            #Change hounsfield units
            data_img[data_img<0] = 0
            data_img[data_img>1000] = 1000
            
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
            
            #Doing padding
            data_img=np.pad(data_img, (pad1, pad2, pad3), constant_values = pad_value)
            data_msk=np.pad(data_msk, (pad1, pad2, pad3), constant_values = pad_value)

            #Find new dimensions
            dim1, dim2, dim3 = data_img.shape
            
            #Define as new Nifti-files
            img_preprocessed = nib.Nifti1Image(data_img, img_resampled_reoriented.affine)
            msk_preprocessed = nib.Nifti1Image(data_msk, msk_resampled_reoriented.affine)
            
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
                for i in range(0,dim1,slice_step):
                    fig, ax = plt.subplots()
                    ax.imshow(data_img[i,:,:].T,cmap="gray",origin="lower")
                    if plot_mask == 1:
                        ax.imshow(data_msk[i,:,:].T,cmap="jet",origin="lower",alpha =0.5*(data_msk[i,:,:].T>0))
                    ax.set_title('Final Dim1, '+str(subject)+", Slice: "+str(i))
                    for v in ctd_resampled_reoriented[1:]:
                        ax.add_patch(Circle((pad2[0]+v[2],pad3[0]+v[3]), 7*1/zooms[0]))
                    plt.show()
                    
            if plot_dim2_after:
                slice_step = int(dim2/Slices)
                if slice_step == 0:
                    slice_step = 1
                for i in range(0,dim2,slice_step):
                    fig, ax = plt.subplots()
                    ax.imshow(data_img[:,i,:].T,cmap="gray",origin="lower")
                    if plot_mask == 1:
                        ax.imshow(data_msk[:,i,:].T,cmap="jet",origin="lower",alpha =0.5*(data_msk[:,i,:].T>0))
                    ax.set_title('Final Dim2, '+str(subject)+", Slice: "+str(i))
                    for v in ctd_resampled_reoriented[1:]:
                        ax.add_patch(Circle((pad1[0]+v[1],pad3[0]+v[3]), 7*1/zooms[2]))
                    plt.show()
            
            if plot_dim3_after:
                slice_step = int(dim3/Slices)
                if slice_step == 0:
                    slice_step = 1
                for i in range(0,dim3,slice_step):
                    fig, ax = plt.subplots()
                    ax.imshow(data_img[:,:,i].T,cmap="gray",origin="lower")
                    if plot_mask == 1:
                        ax.imshow(data_msk[:,i,:].T,cmap="jet",origin="lower",alpha =0.5*(data_msk[:,i,:].T>0))
                    ax.set_title('Final Dim3, '+str(subject)+", Slice: "+str(i))
                    for v in ctd_resampled_reoriented[1:]:
                        ax.add_patch(Circle((pad1[0]+v[1],pad2[0]+v[2]), 7*1/zooms[2]))
                    plt.show()
                
            #Save data
            nib.save(img_preprocessed, os.path.join(Output_folder, subject+'_img.nii.gz'))
            nib.save(msk_preprocessed, os.path.join(Output_folder, subject+'_msk.nii.gz'))
            save_centroids(ctd_resampled_reoriented, os.path.join(Output_folder, subject+'.json'))
            
            
            
    
