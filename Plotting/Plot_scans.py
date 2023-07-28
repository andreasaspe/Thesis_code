import os
import nibabel as nib
import nibabel.orientations as nio
from data_utilities import *
# from nibabel.affines import apply_affine
# import numpy.linalg as npl
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
from my_data_utils import *
from my_plotting_functions import *




#######################################################
#################### CONTROL PANEL ####################
#######################################################
#Define scans
all_scans = 0 #Set to 1 if you want to preprocess all scans
list_of_subjects = ['sub-gl279'] #List of subjects, 'sub-verse510' 500, verse605
#Dårlig opløsning!
#sub-verse510
#sub-verse544
#God opløsning:
#sub-verse807

#Define rescale and reorientation parameters
New_orientation = ('L', 'A', 'S')
New_voxel_size = 8 # [mm]

#Define preprocessing details and printing outputs
Print_info = 1 #Set to 1 to print a lot of info on each scan.

#Define directories
dir_img = '/Users/andreasaspe/Documents/Data/Verse20/Verse20_test_unpacked' #'/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation2/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_training_unpacked' #'/Users/andreasaspe/Documents/Data/Verse20/SpineLocalisation/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_training_unpacked' #'/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_training_unpacked' #'/Users/andreasaspe/Documents/Data/CTSpine1K/trainset/gt' #/Users/andreasaspe/Documents/Data/Verse20/Verse20_test_unpacked' #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data\img' #'/zhome/bb/f/127616/Documents/Thesis/Preprocessed_data'
dir_msk = dir_img #'/Users/andreasaspe/Documents/Data/Verse20/SpineLocalisation/Verse20_training_prep/msk' #'/Users/andreasaspe/Documents/Data/CTSpine1K/trainset/gt' #dir_img #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data\msk'
dir_ctd = dir_img #'/Users/andreasaspe/Documents/Data/Verse20/SpineLocalisation/Verse20_training_prep/ctd' #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data\ctd'

##### PLOTTING ######
Slices = 40 #Slices per volume
plot_scan = 1 #Set to 1 for plotting scan
plot_mask = 1 #Set to 1 for plotting mask
plot_centroids = 1 #Set to 1 for plotting centroids
plot_dim1 = 1
plot_dim2 = 0
plot_dim3 = 0
#######################################################
#######################################################
#######################################################

#Define list of scans
if all_scans:
    all_subjects = []
    for filename in listdir(dir_img):
        subject = filename.split("_")[0]
        if subject.find('.DS') == -1 and subject.find('verse') != -1: #PLOTTER KUN VERSE. IKKE GL
            all_subjects.append(subject)
    all_subjects = np.unique(all_subjects)
else:
    all_subjects = list_of_subjects


zooms_dim1 = []
zooms_dim2 = []
zooms_dim3 = []
data_shape_voxels_dim1 = []
data_shape_voxels_dim2 = []
data_shape_voxels_dim3 = []


#FOR LOOP START
for subject in all_subjects:
    print("       SUBJECT: "+str(subject)+"\n")
    # LOAD FILES
    filename_img = [f for f in listdir(dir_img) if (f.startswith(subject) and f.endswith('img.nii.gz'))][0] #and f.endswith('img.nii.gz'))
    img_nib = nib.load(os.path.join(dir_img,filename_img))
    if plot_mask == 1:
        filename_msk = [f for f in listdir(dir_msk) if (f.startswith(subject) and f.endswith('msk.nii.gz'))][0] #and f.endswith('msk.nii.gz'))
        msk_nib = nib.load(os.path.join(dir_msk,filename_msk))
    if plot_centroids == 1:
        filename_ctd = [f for f in listdir(dir_ctd) if (f.startswith(subject) and f.endswith('.json'))][0] #and f.endswith('.json'))
        ctd_list = load_centroids(os.path.join(os.path.join(dir_ctd,filename_ctd)))
            
    #Get info
    zooms = img_nib.header.get_zooms() #Voxel sizes
    axs_code = nio.ornt2axcodes(nio.io_orientation(img_nib.affine)) #Image orientation
    data_shape_voxels = img_nib.header.get_data_shape() #Shape of data
    data_shape_mm = np.array(data_shape_voxels)*np.array(zooms) #Data measures
    data_type = img_nib.header.get_data_dtype() #Data type

    #Print info
    if Print_info:
        print("Before any preprocessing:")
        print('Zooms: '+str(zooms))
        print('img orientation code: {}'.format(axs_code))
        print('img data shape in voxels: {}'.format(data_shape_voxels))
        print('img data shape in mm: {}'.format(data_shape_mm))
        print('img data type: {}'.format(data_type))
        print("\n")
        
        zooms_dim1.append(zooms[0])
        zooms_dim2.append(zooms[1])
        zooms_dim3.append(zooms[2])
        data_shape_voxels_dim1.append(data_shape_voxels[0])
        data_shape_voxels_dim2.append(data_shape_voxels[1])
        data_shape_voxels_dim3.append(data_shape_voxels[2])
        
        
    #Load data for plotting
    if plot_dim1 or plot_dim2 or plot_dim3:
        img_data = img_nib.get_fdata() #Load data
        #Normalize HU
        # img_data[img_data<-200] = -200
        # img_data[img_data>1000] = 1000
        #Find dimensions
        dim1, dim2, dim3 = img_data.shape #Find dimensions
    if plot_mask == 1:
        msk_data = msk_nib.get_fdata() #Load data
        #msk_data[msk_data != 18] = 0 #If only the bottom 8
    

    #Plot
    if plot_dim1:
        slice_step = int(dim1/Slices)
        if slice_step == 0:
            slice_step = 1
        max_val = img_data.max()
        min_val = img_data.min()
        for i in range(0,dim1,slice_step):
            fig, ax = plt.subplots()
            if plot_scan == 1:
                ax.imshow(img_data[i,:,:].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)            
            if plot_mask == 1:
                ax.imshow(msk_data[i,:,:].T,cmap="jet",origin="lower", alpha =0.99*(msk_data[i,:,:].T>0)) #ax.imshow(msk_data[i,:,:].T,cmap="jet",origin="lower",vmin = 17, vmax = 24, alpha =0.99*(msk_data[i,:,:].T>0))

            if plot_centroids == 1:
                for v in ctd_list[1:]:
                    ax.add_patch(Circle((v[2],v[3]), 7*1/zooms[0])) #7*1/zooms[0]
            ax.set_title('Dim1, '+str(subject)+", Slice: "+str(i))
            plt.axis('off')
            plt.show()
    
    if plot_dim2:
        slice_step = int(dim2/Slices)
        if slice_step == 0:
            slice_step = 1
        max_val = img_data.max()
        min_val = img_data.min()
        for i in range(0,dim2,slice_step):
            fig, ax = plt.subplots()
            if plot_scan == 1:
                ax.imshow(img_data[:,i,:].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)  
            if plot_mask == 1:
                ax.imshow(msk_data[:,i,:].T,cmap="jet",origin="lower",alpha =0.99*(msk_data[:,i,:].T>0)) #Alpha gangedes med 0.5 før
            if plot_centroids == 1:
                for v in ctd_list[1:]:
                    ax.add_patch(Circle((v[1],v[3]), 7*1/zooms[1]))
            ax.set_title('Dim2, '+str(subject)+", Slice: "+str(i))
            plt.axis('off')
            plt.show()

    if plot_dim3:
        slice_step = int(dim3/Slices)
        if slice_step == 0:
            slice_step = 1
        max_val = img_data.max()
        min_val = img_data.min()
        for i in range(0,dim3,slice_step):
            fig, ax = plt.subplots()
            if plot_scan == 1:
                ax.imshow(img_data[:,:,i].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)  
            if plot_mask == 1:
                ax.imshow(msk_data[:,:,i].T,cmap="jet",origin="lower",alpha =0.99*(msk_data[:,:,i].T>0)) #Alpha gangedes med 0.5 før
            if plot_centroids == 1:
                for v in ctd_list[1:]:
                    if v[0] == 22:
                        ax.add_patch(Circle((v[1],v[2]), 7*1/zooms[2]))
            ax.set_title('Dim3, '+str(subject)+", Slice: "+str(i))
            plt.axis('off')
            plt.show()







#SAVE FIG
# if plot_dim1:
#     slice_step = int(dim1/Slices)
#     if slice_step == 0:
#         slice_step = 1
#     max_val = img_data.max()
#     min_val = img_data.min()
#     for i in range(0,dim1,slice_step):
#         fig, ax = plt.subplots()
#         if plot_scan == 1:
#             ax.imshow(img_data[i,:,:].T,cmap="gray",origin="lower",vmin = min_val, vmax = max_val)            
#         if plot_mask == 1:
#             ax.imshow(msk_data[i,:,:].T,cmap="jet",origin="lower",vmin = 17, vmax = 24, alpha = 0.99*(msk_data[i,:,:].T>0))
#             if i == 240:
#                 plt.savefig('/Users/andreasaspe/Desktop/heatmap',dpi=500,transparent=True)
#         if plot_centroids == 1:
#             for v in ctd_list[1:]:
#                 ax.add_patch(Circle((v[2],v[3]), 7*1/zooms[0]))
#         ax.set_title('Dim1, '+str(subject)+", Slice: "+str(i))
#         #plt.axis('off')
#         plt.show()