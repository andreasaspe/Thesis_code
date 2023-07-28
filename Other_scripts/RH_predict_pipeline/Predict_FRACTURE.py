# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 17:12:09 2023

@author: PC
"""

import cc3d
import numpy as np
import nibabel as nib
from tqdm import tqdm
import os
from os import listdir
import pickle
import matplotlib.pyplot as plt
from skimage.measure import label
import copy
import scipy #For 2D connected components
import cv2 #For contours
from copy import deepcopy
import time
import pandas as pd
#My imports
from my_plotting_functions import *
from my_data_utils import *
import math



#######################################################
#################### CONTROL PANEL ####################
#######################################################
#Define scans
all_scans = 1
list_of_subjects = ['VERTEBRAE_FRACTURE_0206_SERIES0005'] #'VERTEBRAE_FRACTURE_0294_SERIES0019'] #['VERTEBRAE_FRACTURE_0206_SERIES0005'] #['VERTEBRAE_FRACTURE_0208_SERIES0007'] #['VERTEBRAE_FRACTURE_0280_SERIES0007'] #'VERTEBRAE_FRACTURE_0208_SERIES0007']#,'VERTEBRAE_FRACTURE_0294_SERIES0019']#'VERTEBRAE_FRACTURE_0334_SERIES0010'] #['VERTEBRAE_FRACTURE_0208_SERIES0007'] #['VERTEBRAE_FRACTURE_0213_SERIES0008','VERTEBRAE_HEALTHY_0000_SERIES0003','VERTEBRAE_FRACTURE_0208_SERIES0007'] #VERTEBRAE_FRACTURE_0239_SERIES0003'] #['VERTEBRAE_HEALTHY_0001_SERIES0010'] #List of subjects
# with open("E:\s174197\Thesis\My_code\Other_scripts\list_of_subjects_FRACTURE", "rb") as fp:   # Unpickling
#     list_of_subjects = pickle.load(fp)

#Oprindelig for at rotere firkant
    # VERTEBRAE_FRACTURE_0280_SERIES0007
    # Eller denne: VERTEBRAE_FRACTURE_0294_SERIES0019. Men den ændrede sig..
    # Nu er det denne: VERTEBRAE_FRACTURE_0212_SERIES0012
#Oprindelig:
    # VERTEBRAE_FRACTURE_0206_SERIES0005
#Den her har en ret sjov først vertebra. Tror der går noget galt:
    # VERTEBRAE_FRACTURE_0208_SERIES0007
    
# VERTEBRAE_FRACTURE_0285_SERIES0015 går den her lidt galt i SpineLocalisation?

#Define directories
dir_segmentations = r'E:\s174197\data_RH\Predictions\Segmentations_afterCCA'
dir_img = 'E:/s174197/data_RH/VertebraeSegmentation_newaffine/data_prep/img'
dir_centroids = r'E:\s174197\data_RH\Predictions\Centroids'
dir_predictions = 'E:/s174197/data_RH/Predictions'
#######################################################
#######################################################
#######################################################


plt.style.use('default')
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12  # Default text - så fx. akse-tal osv.
plt.rcParams["axes.titlesize"] = 12  # Size for titles
plt.rcParams["axes.labelsize"] = 15  # Size for labels
plt.rcParams["legend.fontsize"] = 12  # Size for legends
plt.rcParams["figure.figsize"] = (6.4, 2) #Standard. Målt i inches. Width x height



#Define list of scans
if all_scans:
    all_subjects = []
    for filename in listdir(dir_segmentations):
        subject = filename.split("-")[0]
        # if subject.find('FRACTURE') != -1: #PLOTTER KUN VERSE. IKKE GL
        all_subjects.append(subject)
    all_subjects = np.unique(all_subjects)
    #Sorterer fil '.DS' fra
    all_subjects = all_subjects[all_subjects != '.DS']
else:
    all_subjects = list_of_subjects
    
# from skimage import morphology
    
# # https://stackoverflow.com/questions/56938207/how-to-remove-small-objects-from-3d-image
# def remove_small_objects(img):
#     binary = copy.copy(img)
#     binary[binary>0] = 1
#     labels = morphology.label(binary)
#     labels_num = [len(labels[labels==each]) for each in np.unique(labels)]
#     rank = np.argsort(np.argsort(labels_num))
#     index = list(rank).index(len(rank)-2)
#     new_img = copy.copy(img)
#     new_img[labels!=index] = 0
#     return new_img
    

# v_dict = {
#     1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7',
#     8: 'T1', 9: 'T2', 10: 'T3', 11: 'T4', 12: 'T5', 13: 'T6', 14: 'T7',
#     15: 'T8', 16: 'T9', 17: 'T10', 18: 'T11', 19: 'T12', 20: 'L1',
#     21: 'L2', 22: 'L3', 23: 'L4', 24: 'L5', 25: 'L6', 26: 'Sacrum',
#     27: 'Cocc', 28: 'T13'
# }


df = pd.read_excel(r'G:/DTU-Vertebra-1/Metadata/DTU-vertebra-1-clinical-data.xlsx')

grades_full_list_GT = []
grades_full_list_pred = []

for subject in tqdm(all_subjects):
    
    #Get fracture information from meta_datafile
    temp = subject.split("_") #Split into substrings
    subject_without_ending = "_".join(temp[:-1]) #Subject name without SERIES. So we have VERTEBRAE_LOWHU_0100_SERIES0018 -> VERTEBRAE_LOWHU_0100

    filtered_df = df[df['ID'] == subject_without_ending]
    
    fractureT10 = filtered_df['ct_vertebra_frac_th10_grade'].values[0]
    fractureT11 = filtered_df['ct_vertebra_frac_th11_grade'].values[0]
    fractureT12 = filtered_df['ct_vertebra_frac_th12_grade'].values[0]
    fractureL1 = filtered_df['ct_vertebra_frac_l1_grade'].values[0]
    fractureL2 = filtered_df['ct_vertebra_frac_l2_grade'].values[0]
    fractureL3 = filtered_df['ct_vertebra_frac_l3_grade'].values[0]
    fractureL4 = filtered_df['ct_vertebra_frac_l4_grade'].values[0]
    fractureL5 = filtered_df['ct_vertebra_frac_l5_grade'].values[0]
    
    fracture_list = [fractureT10,fractureT11,fractureT12,fractureL1,fractureL2,fractureL3,fractureL4,fractureL5]
    grades_list_GT = []
    grades_list_pred = [-1, -1, -1, -1, -1, -1, -1, -1]
    
    for v_fracture in fracture_list:
        if str(v_fracture).find('Grade') != -1:
            grade_GT = int(v_fracture[6])
        else:
            grade_GT = 0
        grades_list_GT.append(grade_GT)

    
    
    print(subject)
    
    # print(mapping_FRACTURE[subject])
    # continue
    
    #Load centroids
    # ctd_list = np.load(os.path.join(dir_centroids,subject+'-centroids.npy'))
    # with open(os.path.join(dir_centroids,subject+'-centroids'), "wb") as fp:   #Pickling
    #     pickle.dump(ctd_list, fp)
    with open(os.path.join(dir_centroids,subject+'-centroids'), 'rb') as file:
        ctd_list = pickle.load(file)
    
    #Load mask
    msk_nib = nib.load(os.path.join(dir_segmentations,subject+'-PREDICTIONafter.nii.gz'))
    mask = np.asanyarray(msk_nib.dataobj, dtype=np.float32)    
    
    #Load image for plotting
    filename_img = subject + "-img.nii.gz"
    img_nib = nib.load(os.path.join(dir_img,filename_img))
    data_img = np.asanyarray(img_nib.dataobj, dtype=np.float32)
    dim1,dim2,dim3 = data_img.shape
    
    # show_mask_dim1(mask,subject)
    # show_mask_img_dim1(data_img,mask,subject)

    N_vertebrae = len(ctd_list)-1 #Minus 1 because of ('L','A','S')
        
    data_img_original = deepcopy(data_img)
    #Isolate vertebrae
    for ctd in ctd_list[1:]: #Fordi vi vil ikke have 0 med, som er baggrund
        
        #Find central slice!
        centroid = (ctd[1], ctd[2], ctd[3])
        
        vertebra = int(ctd[0])
        
        # if vertebra != 22:
        #     continue
         
        #Find coordinates
        x_coordinate = int(np.round(centroid[0]))
        y_coordinate = int(np.round(centroid[1]))
        z_coordinate = int(np.round(centroid[2]))
        
        #Crop central slice
        V = mask[x_coordinate,:,:]
        data_img = data_img_original[x_coordinate,:,:]
        # show_one_slice(V,subject)

        
        #Only set relevant vertebra to 1 and the rest to 0
        V = np.where(V == vertebra,1,0) #V for vertebrae
        
        show_one_mask_img_slice(data_img,V,subject)

        
        #Find row and column indicies for true values (1s) - the mask
        y_indices, z_indices = np.where(V != 0)
        
        # Calculate the bounding box coordinates
        y_min = np.min(y_indices)
        y_max = np.max(y_indices)
        z_min = np.min(z_indices)
        z_max = np.max(z_indices)
        
        #Crop tightly to vertebra
        V = V[y_min-20:y_max+1+20,max(0,z_min-20):z_max+1+20]
        data_img = data_img[y_min:y_max+1,z_min:z_max+1]
            
        #Find blobs for sorting out the posterior part of vertebra
        blobs, no_blobs = scipy.ndimage.label(V)
        
        # show_one_slice(blobs,subject)
        # show_one_slice(V,subject)

        #Number of blobs. This should be 2.
        no_unique = len(np.unique(blobs)) #blobs er labelled array. no_blobs er antal blobs i den array.
        
        #Define empty list
        max_y_coordinates = []
        
        # #Get the y-coordinates
        # for label in range(1,no_unique):
        #     max_y_coordinates.append(np.max(np.where(blobs == label)[0]))
        
        # #Get the index with the blobs with coordinates furtherst to the right (the vertebral body)
        # Highest_blob_index = np.argmax(max_y_coordinates)+1 #Fordi jeg sortede 0 fra før oppe i forløbet. Se mit range. 0 er baggrunds-class
        areas_list = []
        
        for label in range(no_unique):
            area = np.count_nonzero(blobs == label)
            areas_list.append(area)
        
        #Get second largest index
        areas_list = np.array(areas_list)
        sorted_indices = np.argsort(-areas_list)
        body_index = sorted_indices[1]

        #Remove everything else than the body
        V_body = np.where(blobs == body_index,1,0)
        
        # show_one_slice(V_body,subject)
        
        #Crop again tightly to the body
        y_indices, z_indices = np.where(V_body != 0)
        
        # Calculate the bounding box coordinates
        y_min = np.min(y_indices)
        y_max = np.max(y_indices)
        z_min = np.min(z_indices)
        z_max = np.max(z_indices)
        
        #Crop
        V_body = V_body[y_min:y_max+1,z_min:z_max+1]
        data_img = data_img[y_min:y_max+1,z_min:z_max+1]
        # show_one_slice(V_body,subject)
        # show_one_slice(data_img,subject)
        # show_one_mask_img_slice(data_img,V_body,subject)
        # break
        
        #Find contours!
        
        contours, _ = cv2.findContours(V_body.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        contour_image = np.zeros_like(V_body, dtype=np.uint8)
        
        # Draw the contours on the new image
        cv2.drawContours(contour_image, contours, -1, 1, thickness=1)
        
        # show_one_slice(contour_image,subject)
        
        
        
        #NEW
        # Assume the largest contour is the rectangular shape
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Compute the minimum area bounding rectangle
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Calculate the angle of rotation
        angle = rect[-1]
        
        # print(angle)
        
        if angle > 45:
            # Rotate the image to make the rectangle horizontal
            center = tuple(np.mean(box, axis=0).astype(int))
            center = (int(center[0]),int(center[1]))
            rotation_matrix = cv2.getRotationMatrix2D(center, -(90-angle), 1.0)
            
            V_body2 = deepcopy(V_body.astype(np.uint8))            
            
            rotated_image = cv2.warpAffine(V_body2, rotation_matrix, (V_body2.shape[1], V_body2.shape[0]))
        else:
            # Rotate the image to make the rectangle horizontal
            center = tuple(np.mean(box, axis=0).astype(int))
            center = (int(center[0]),int(center[1]))
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            V_body2 = deepcopy(V_body.astype(np.uint8))            
                        
            rotated_image = cv2.warpAffine(V_body2, rotation_matrix, (V_body2.shape[1], V_body2.shape[0]))
            
            # contour_image = np.zeros_like(V_body, dtype=np.uint8)
            # # Draw the contours on the new image
            # cv2.drawContours(contour_image, contours, -1, 1, thickness=1)
            # rotated_image = contour_image
            
        # show_one_slice(rotated_image,'')
        
        
        
        V_body = rotated_image
        
        contours, _ = cv2.findContours(V_body.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        ###################################################
        # Create a new blank image to draw the contours
        contour_image = np.zeros_like(V_body, dtype=np.uint8)
        
        # Draw the contours on the new image
        cv2.drawContours(contour_image, contours, -1, 1, thickness=1)

        # show_one_slice(contour_image,subject)
        # time.sleep(0.5)
        # continue


        dim1, dim2 = V_body.shape
       
        
        # if vertebra != 24:
        #     continue
        
        #Initialise
        distances_list = []
        pixels_list = []
        
        #Screening body
        i=0
        while i+1 <= dim1:
            # V_temp = contour_image[i:i+1,:]
            V_temp = contour_image[i:i+1,:]
            
            # show_one_slice(V_temp,subject)
            # show_one_slice(Upper_half,subject)
            # show_one_slice(Lower_haft,subject)
            
            half_point = int(np.round(dim2/2))
            
            #Lower lower half
            Lower_half = V_temp[:,0:half_point]
            y_indices_lower, z_indices_lower = np.where(Lower_half == 1)
            
            
            for pixel_val in np.unique(y_indices_lower):
                index_y = np.where(y_indices_lower == pixel_val)[0]
                if len(index_y) > 1:
                    index_minval = np.argmin(z_indices_lower[index_y])
                    indices_to_delete = np.delete(index_y,index_minval)
                    # for mm in indices_to_delete:
                    #     Lower_half[pixel_val,z_indices_lower[mm]] = 0
                    y_indices_lower = np.delete(y_indices_lower,indices_to_delete)
                    z_indices_lower = np.delete(z_indices_lower,indices_to_delete)
            
            #Upper lower half
            Upper_half = V_temp[:,half_point:dim2]
            y_indices_upper, z_indices_upper = np.where(Upper_half == 1)
            

            for pixel_val in np.unique(y_indices_upper):
                index_y = np.where(y_indices_upper == pixel_val)[0]
                if len(index_y) > 1:
                    index_maxval = np.argmax(z_indices_upper[index_y])
                    indices_to_delete = np.delete(index_y,index_maxval)
                    # for mm in indices_to_delete:
                    #     Upper_half[pixel_val,z_indices_upper[mm]] = 0
                    y_indices_upper = np.delete(y_indices_upper,indices_to_delete)
                    z_indices_upper = np.delete(z_indices_upper,indices_to_delete)
                    
            
            #Adjust for cropping:
            z_indices_upper+=half_point
            y_indices_lower+=i
            y_indices_upper+=i
            
            #Check if they are empty
            if len(y_indices_lower) == 0 or len(y_indices_upper) == 0:
                i+=1
                continue
            
            x1 = y_indices_lower[0]
            x2 = y_indices_upper[0]
            y1 = z_indices_lower[0]
            y2 = z_indices_upper[0]
            

            dist = y2-y1
           
            pixels = (x1,y1,x2,y2)
           
            if dist < 15:
                pass
            else:
                distances_list.append(dist)
                pixels_list.append(pixels)
                # show_one_slice_with_pixel_markings(data = V_body, pixels = pixels_lower, subject=subject, color = 'blue')
           
            i+=1
           
           
        # All pixels!
        # for pixels in pixels_list:
        #     show_one_slice_with_pixel_markings(data = V_body, pixels = pixels, subject=subject, color = 'grey')
            
        
 
        # break
        if len(pixels_list) == 0:
            print("Something is wrong with {}. Cannot estimate fracture".format(my_v_dict[vertebra+1]))
            continue
        else:
            pixels_min_list = deepcopy(pixels_list)
            pixels_max_list = deepcopy(pixels_list)
            distances_min_list = deepcopy(distances_list)
            distances_max_list = deepcopy(distances_list)
       
        def is_it_a_full_column(array):
            occurrences = 0 
            process_started = 0
            len_of_chain = 0
            for i in range(len(array)-1):
                if array[i] == 1:
                    
                    process_started = 1
                    len_of_chain+=1
                    
                    if occurrences == 1: #Så betyder det, at den allerede har set den én gang.
                        return False, 0
                else:
                    if process_started == 1:
                        occurrences=1
                    else:
                        pass
            return True, len_of_chain
        
        def find_longest_chain(array):
            unique_ID = 0 
            process_started = 0
            chain = []
            len_of_chain = []
            for i in range(len(array)):
                if array[i] == 1:
                    chain.append(unique_ID)
                else:
                    unique_ID+=1
            chain = np.array(chain)
            for unique_val in np.unique(chain):
                len_of_chain.append(sum(chain==unique_val))
            return max(len_of_chain)
        
        
        
        #Remove last elements in list MIN
        stop=False
        i=1
        distances_min_list_original = deepcopy(distances_min_list)
        pixels_min_list_original = deepcopy(pixels_min_list)
        while not stop:
            pixel = pixels_min_list_original[-i]
            firstpixel = pixel[0]
            
            firstcolumn = V_body[firstpixel,:]
            
            len_of_chain1 = find_longest_chain(firstcolumn)
            
            if len_of_chain1 > 10:
                stop = True
            else:
                stop = False
                #Pop elements
                distances_min_list.pop()
                pixels_min_list.pop()
            #Update counter
            i+=1
            
        #Remove last elements in list MAX
        stop=False
        i=1
        distances_max_list_original = deepcopy(distances_max_list)
        pixels_max_list_original = deepcopy(pixels_max_list)
        while not stop:
            pixel = pixels_max_list_original[-i]
            firstpixel = pixel[0]
            
            firstcolumn = V_body[firstpixel,:]
            
            len_of_chain1 = find_longest_chain(firstcolumn)
            
            if len_of_chain1 > 10:
                stop = True
            else:
                stop = False
                #Pop elements
                distances_max_list.pop()
                pixels_max_list.pop()
            #Update counter
            i+=1
            
            
            
        #Check for too large jumps in the end
        stop=False
        i=1
        # distances_min_list_original = copy.copy(distances_min_list)
        # pixels_min_list_original = copy.copy(pixels_min_list)
        while not stop:
            if abs(distances_min_list[-i]-distances_min_list[-(i+1)]) >= 1:
                stop = False
                #Pop elements
                distances_min_list.pop()
                pixels_min_list.pop()
                #Update counter
                i+=1
            else:
                stop = True
                
                
                
            
        #Remove first elements in list MIN
        stop=False
        i=0
        distances_min_list_original2 = copy.copy(distances_min_list)
        pixels_min_list_original2 = copy.copy(pixels_min_list)
        while not stop:
            pixel = pixels_min_list_original2[i]
            firstpixel = pixel[0]
            
            firstcolumn = V_body[firstpixel,:]
            
            len_of_chain1 = find_longest_chain(firstcolumn)
            
            if len_of_chain1 > 10:
                stop = True
            else:
                stop = False
                #Pop elements
                distances_min_list.pop(0)
                pixels_min_list.pop(0)
            #Update counter
            i+=1
            
            
        #Remove first elements in list MAX
        stop=False
        i=0
        distances_max_list_original2 = copy.copy(distances_max_list)
        pixels_max_list_original2 = copy.copy(pixels_max_list)
        while not stop:
            pixel = pixels_max_list_original2[i]
            firstpixel = pixel[0]
            
            firstcolumn = V_body[firstpixel,:]
            
            len_of_chain1 = find_longest_chain(firstcolumn)
            
            if len_of_chain1 > 10:
                stop = True
            else:
                stop = False
                #Pop elements
                distances_max_list.pop(0)
                pixels_max_list.pop(0)
            #Update counter
            i+=1
        
            
            
        #Check for too large jumps in the beginning
        stop=False
        i=0
        # distances_min_list_original = copy.copy(distances_min_list)
        # pixels_min_list_original = copy.copy(pixels_min_list)
        while not stop:
            if abs(distances_min_list[i]-distances_min_list[i+1]) >= 1:
                stop = False
                #Pop elements
                distances_min_list.pop(0)
                pixels_min_list.pop(0)
                #Update counter
                i+=1
            else:
                stop = True
                
                
        #Largest distance must be in on of the two ends for the long part
        filtering = np.ones(len(distances_max_list),dtype=bool)
        filtering[6:-6] = False
        distances_max_list = np.array(distances_max_list)[filtering]
        pixels_max_list = np.array(pixels_max_list)[filtering]
                    
            
        # Min pixels!
        # for pixels in pixels_min_list:
        #     show_one_slice_with_pixel_markings(data = V_body, pixels = pixels, subject=subject, color = 'blue')
            
        # # # # Max pixels!
        # for pixels in pixels_max_list:
        #     show_one_slice_with_pixel_markings(data = V_body, pixels = pixels, subject=subject, color = 'red')
        
                
 
        #Find out if there is a fracture
        min_arg_dist = np.argmin(distances_min_list)
        max_arg_dist = np.argmax(distances_max_list)
        
        min_pixel = pixels_min_list[min_arg_dist]
        max_pixel = pixels_max_list[max_arg_dist]
        
        #Sorted_lists
        min_dist_argsort = np.argsort(distances_min_list)
        max_dist_argsort = np.flip(np.argsort(distances_max_list))
        
        pixels_in_scope = {}
        for j in range(5): #j is max index, note they are interchanged. Does not mean anything. Only for the ordering
            for i in range(5): #i is min index. Does not mean anything. Only for the ordering.
                min_pixel = pixels_min_list[min_dist_argsort[i]]
                x_min = min_pixel[0]
                
                max_pixel = pixels_max_list[max_dist_argsort[j]]
                x_max = max_pixel[0]
                
                if abs(x_max-x_min) > 7: #Minimum 5 pixels in difference!
                    min_dist = distances_min_list[min_dist_argsort[i]]
                    max_dist = distances_max_list[max_dist_argsort[j]]
                    
                    reduction = ((max_dist-min_dist)/max_dist)*100

                    pixels_in_scope[(i,j)] = reduction

        reduction = max(pixels_in_scope.values()) #Maximum reduction!
        key = next(key for key, value in pixels_in_scope.items() if value == reduction)
        
        min_idx = key[0] #Get the i index
        max_idx = key[1] #Get the j index
        
        min_pixel = pixels_min_list[min_dist_argsort[min_idx]]
        max_pixel = pixels_max_list[max_dist_argsort[max_idx]]

        show_one_slice_with_two_markings(V_body, min_pixel, max_pixel, 'blue', 'red', subject)
        # show_one_slice_with_two_markings(contour_image, min_pixel, max_pixel, 'blue', 'red', subject)
        
        #Show distance plot!
        # start_pixel = pixels_list[0][0]
        # end_pixel = pixels_list[-1][0]
        # x = np.arange(start_pixel,end_pixel+1)
        # y = distances_list
        # plt.figure()
        # plt.plot(x,y,linewidth=2,color='black')
        # plt.scatter(min_pixel[0],min_pixel[3]-min_pixel[1], c='blue',s=100,zorder=2)
        # plt.scatter(max_pixel[0],max_pixel[3]-max_pixel[1], c='red',s=100, zorder=2)
        # # plt.axis('off')
        # plt.xlabel('Vertebral body length')
        # plt.ylabel('Distance')
        # plt.box(False)
        # plt.grid(True)
        # plt.xticks([])
        # plt.yticks([])
        # plt.show()


        if reduction < 20:
            grade_pred = 0
            print("\nThere is no fracture in {}. Reduction is {}".format(number_to_name[vertebra],np.round(reduction,1)))
        elif 20 <= reduction < 25:
            grade_pred = 1
            print("\nFracture of grade 1 in {} detected! Reduction is {}".format(number_to_name[vertebra],np.round(reduction,1)))
        elif 25 <= reduction < 40:
            grade_pred = 2
            print("\nFracture of grade 2 in {} detected! Reduction is {}".format(number_to_name[vertebra],np.round(reduction,1)))
        elif reduction >= 40:
            grade_pred = 3
            print("\nFracture of grade 3 in {} detected! Reduction is {}".format(number_to_name[vertebra],np.round(reduction,1)))
        else:
            grade_pred = -1
            print("\nSomething went wrong with {}. Reduction is {}".format(number_to_name[vertebra],np.round(reduction,1)))
           
        #Update list
        grades_list_pred[vertebra-17] = grade_pred
    # break
        
    grades_full_list_pred.append(grades_list_pred)
    grades_full_list_GT.append(grades_list_GT)
            
            

    
column_names = ['T10','T11','T12','L1','L2','L3','L4','L5']

df_pred = pd.DataFrame(grades_full_list_pred, columns=column_names)
df_pred.insert(0, 'subjects', all_subjects)
df_pred.to_csv(os.path.join(dir_predictions,'Fracture_pred.csv'), index=False)

df_GT = pd.DataFrame(grades_full_list_GT, columns=column_names)
df_GT.insert(0, 'subjects', all_subjects)
df_GT.to_csv(os.path.join(dir_predictions,'Fracture_GT.csv'), index=False)

            
     
            
            
            

# df_stage1 = pd.DataFrame({'subjects': subject_data, 'mse': mse_data,'iou':iou_list})
# df_stage1 = df_stage1.sort_values(by='mse', ascending=True)
# df_stage1.to_csv(os.path.join(dir_predictions_dataframe_folder,'df_stage1_training.csv'), index=False)

                

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        #     for ii in range(len(y_indices_lower)):
        #         for jj in range(len(z_indices_upper)):
        #             x1, y1 = y_indices_lower[ii], z_indices_lower[ii]
        #             x2, y2 = y_indices_upper[jj], z_indices_upper[jj]
        #             dist = y2-y1
                    
    
        #     # Find the key with the lowest value
        #     min_value = min(distances_temp.values())
        #     min_key = next(key for key, value in distances_temp.items() if value == min_value)
        #     min_y_lower = y_indices_lower[min_key[0]]
        #     min_z_lower = z_indices_lower[min_key[0]]
        #     min_y_upper = y_indices_upper[min_key[1]]
        #     min_z_upper = z_indices_upper[min_key[1]]
            
        #     pixels_lower = (min_y_lower,min_z_lower,min_y_upper,min_z_upper)
            
        #     if min_value < 20:
        #         pass
        #     else:
        #         distances_min_list.append(min_value)
        #         pixels_min_list.append(pixels_lower)
        #         # show_one_slice_with_pixel_markings(data = V_body, pixels = pixels_lower, subject=subject, color = 'blue')
            
        #     max_value = max(distances_temp.values())
        #     max_key = next(key for key, value in distances_temp.items() if value == max_value)
        #     max_y_lower = y_indices_lower[max_key[0]]
        #     max_z_lower = z_indices_lower[max_key[0]]
        #     max_y_upper = y_indices_upper[max_key[1]]
        #     max_z_upper = z_indices_upper[max_key[1]]
            
        #     pixels_upper = (max_y_lower,max_z_lower,max_y_upper,max_z_upper)
            
        #     if max_value < 20:
        #         pass
        #     else:
        #         distances_max_list.append(max_value)
        #         pixels_max_list.append(pixels_upper)
        #         # show_one_slice_with_pixel_markings(data = V_body, pixels = pixels_upper, subject=subject, color = 'red')
            
        #     # show_one_slice_with_pixel_markings(data = V_body, pixels = pixels_lower, subject=subject, color = 'blue')
        #     # show_one_slice_with_pixel_markings(data = V_body, pixels = pixels_upper, subject=subject, color = 'red')
            
        #     #Update counter
        #     i+=1
            
            
        # # Min pixels!
        # for pixels in pixels_min_list:
        #     show_one_slice_with_pixel_markings(data = V_body, pixels = pixels, subject=subject, color = 'blue')
           
        # # break
        # #Max pixels!
        # for pixels in pixels_max_list:
        #     show_one_slice_with_pixel_markings(data = V_body, pixels = pixels, subject=subject, color = 'red')
                
        # break
        
        # def is_it_a_full_column(array):
        #     occurrences = 0 
        #     process_started = 0
        #     len_of_chain = 0
        #     for i in range(len(array)-1):
        #         if array[i] == 1:
                    
        #             process_started = 1
        #             len_of_chain+=1
                    
        #             if occurrences == 1: #Så betyder det, at den allerede har set den én gang.
        #                 return False, 0
        #         else:
        #             if process_started == 1:
        #                 occurrences=1
        #             else:
        #                 pass
        #     return True, len_of_chain
        
        # def find_longest_chain(array):
        #     unique_ID = 0 
        #     process_started = 0
        #     chain = []
        #     len_of_chain = []
        #     for i in range(len(array)):
        #         if array[i] == 1:
        #             chain.append(unique_ID)
        #         else:
        #             unique_ID+=1
        #     chain = np.array(chain)
        #     for unique_val in np.unique(chain):
        #         len_of_chain.append(sum(chain==unique_val))
        #     return max(len_of_chain)
        
        
        # #Remove last elements in list
        # stop=False
        # i=1
        # distances_min_list_original = copy.copy(distances_min_list)
        # pixels_min_list_original = copy.copy(pixels_min_list)
        # while not stop:
        #     pixel = pixels_min_list_original[-i]
        #     firstpixel = pixel[0]
        #     secondpixel = pixel[2]
            
        #     firstcolumn = V_body[firstpixel,:]
        #     secondcolumn = V_body[secondpixel,:]
            
            
        #     is_first_full,len_of_chain1 = is_it_a_full_column(firstcolumn)
        #     is_second_full,len_of_chain2 = is_it_a_full_column(secondcolumn)
            
            
        #     if is_first_full or is_second_full:
        #         if len_of_chain1 > 15 or len_of_chain2 > 15:
        #             stop = True
        #         else:
        #             stop = False
        #     else:
        #         stop = False
        #     #Pop elements
        #     distances_min_list.pop()
        #     pixels_min_list.pop()
        #     #Update counter
        #     i+=1
            
            
        #     # len_of_chain1 = find_longest_chain(firstcolumn)
        #     # len_of_chain2 = find_longest_chain(secondcolumn)
            
        #     # if len_of_chain1 > 15 or len_of_chain2 > 15:
        #     #     stop = True
        #     # else:
        #     #     stop = False
        #     # #Pop elements
        #     # distances_min_list.pop()
        #     # pixels_min_list.pop()
        #     # #Update counter
        #     # i+=1
            
            
        #     # is_first_full,len_of_chain1 = is_it_a_full_column(firstcolumn)
        #     # is_second_full,len_of_chain2 = is_it_a_full_column(secondcolumn)
        
        #     # if is_first_full or is_second_full:
        #     #     if len_of_chain1 > 15 or len_of_chain2 > 15:
        #     #         stop = True
        #     #     else:
        #     #         stop = False
        #     # else:
        #     #     stop = False
        #     # #Pop elements
        #     # distances_min_list.pop()
        #     # pixels_min_list.pop()
        #     # #Update counter
        #     # i+=1
            
            
        # #Check for too large jumps in the end
        # stop=False
        # i=1
        # # distances_min_list_original = copy.copy(distances_min_list)
        # # pixels_min_list_original = copy.copy(pixels_min_list)
        # while not stop:
        #     if abs(distances_min_list[-i]-distances_min_list[-(i+1)]) >= 3:
        #         stop = False
        #         #Pop elements
        #         distances_min_list.pop()
        #         pixels_min_list.pop()
        #         #Update counter
        #         i+=1
        #     else:
        #         stop = True
                
                
                
            
        # #Remove first elements in list
        # stop=False
        # i=0
        # distances_min_list_original2 = copy.copy(distances_min_list)
        # pixels_min_list_original2 = copy.copy(pixels_min_list)
        # while not stop:
        #     pixel = pixels_min_list_original2[i]
        #     firstpixel = pixel[0]
        #     secondpixel = pixel[2]
            
        #     firstcolumn = V_body[firstpixel,:]
        #     secondcolumn = V_body[secondpixel,:]
            
        #     len_of_chain1 = find_longest_chain(firstcolumn)
        #     len_of_chain2 = find_longest_chain(secondcolumn)
            
        #     if len_of_chain1 > 15 or len_of_chain2 > 15:
        #         stop = True
        #     else:
        #         stop = False
        #     #Pop elements
        #     distances_min_list.pop(0)
        #     pixels_min_list.pop(0)
        #     #Update counter
        #     i+=1
            
            
        #     # is_first_full,len_of_chain1 = is_it_a_full_column(firstcolumn)
        #     # is_second_full,len_of_chain2 = is_it_a_full_column(secondcolumn)
            
            
        #     # if is_first_full or is_second_full:
        #     #     if len_of_chain1 > 15 or len_of_chain2 > 15:
        #     #         stop = True
        #     #     else:
        #     #         stop = False
        #     # else:
        #     #     stop = False
        #     # #Pop elements
        #     # distances_min_list.pop(0)
        #     # pixels_min_list.pop(0)
        #     # #Update counter
        #     # i+=1
            
        # #Check for too large jumps in the beginning
        # stop=False
        # i=0
        # # distances_min_list_original = copy.copy(distances_min_list)
        # # pixels_min_list_original = copy.copy(pixels_min_list)
        # while not stop:
        #     if abs(distances_min_list[i]-distances_min_list[i+1]) >= 2:
        #         stop = False
        #         #Pop elements
        #         distances_min_list.pop(0)
        #         pixels_min_list.pop(0)
        #         #Update counter
        #         i+=1
        #     else:
        #         stop = True
                
                
        # #Largest distance must be in on of the two ends for the long part
        # filtering = np.ones(len(distances_max_list),dtype=bool)
        # filtering[6:-6] = False
        # distances_max_list = np.array(distances_max_list)[filtering]
        # pixels_max_list = np.array(pixels_max_list)[filtering]
                    
            
        # # # Min pixels!
        # # for pixels in pixels_min_list:
        # #     show_one_slice_with_pixel_markings(data = V_body, pixels = pixels, subject=subject, color = 'blue')
            
        # # # # # Max pixels!
        # # for pixels in pixels_max_list:
        # #     show_one_slice_with_pixel_markings(data = V_body, pixels = pixels, subject=subject, color = 'red')
                  
                

        # #Find out if there is a fracture
        # min_arg_dist = np.argmin(distances_min_list)
        # max_arg_dist = np.argmax(distances_max_list)
        
        # min_pixel = pixels_min_list[min_arg_dist]
        # max_pixel = pixels_max_list[max_arg_dist]
        
        # show_one_slice_with_pixel_markings(data = V_body, pixels = min_pixel, subject=subject, color = 'blue')
        # show_one_slice_with_pixel_markings(data = V_body, pixels = max_pixel, subject=subject, color = 'red')
        
        
        # min_dist = np.min(distances_min_list)
        # max_dist = np.max(distances_max_list)
        
        # reduction = ((max_dist-min_dist)/max_dist)*100
        
        # if reduction < 20:
        #     print("\nThere is no fracture in {}. Reduction is {}".format(my_v_dict[vertebra+1],np.round(reduction,1)))
        # elif 20 <= reduction < 25:
        #     print("\nFracture of grade 1 in {} detected! Reduction is {}".format(my_v_dict[vertebra+1],np.round(reduction,1)))
        # elif 25 <= reduction < 40:
        #     print("\nFracture of grade 2 in {} detected! Reduction is {}".format(my_v_dict[vertebra+1],np.round(reduction,1)))
        # elif reduction >= 40:
        #     print("\nFracture of grade 3 in {} detected! Reduction is {}".format(my_v_dict[vertebra+1],np.round(reduction,1)))
        # else:
        #     print("\nSomething went wrong with {}. Reduction is {}".format(my_v_dict[vertebra+1],np.round(reduction,1)))
            























































#OLD:
        # dim1, dim2 = V_body.shape
       
        
        # # if vertebra != 4:
        # #     continue
        
        # #Initialise
        # distances_min_list = []
        # pixels_min_list = []
        # distances_max_list = []
        # pixels_max_list = []
        
        # #Screening body
        # i=0
        # while i+1 <= dim1:
        #     # V_temp = contour_image[i:i+1,:]
        #     V_temp = contour_image[i:i+4,:]
            
        #     # show_one_slice(V_temp,subject)
        #     # show_one_slice(Upper_half,subject)
        #     # show_one_slice(Lower_haft,subject)
            
        #     half_point = int(np.round(dim2/2))
            
        #     #Lower lower half
        #     Lower_half = V_temp[:,0:half_point]
        #     y_indices_lower, z_indices_lower = np.where(Lower_half == 1)
            
            
        #     for pixel_val in np.unique(y_indices_lower):
        #         index_y = np.where(y_indices_lower == pixel_val)[0]
        #         if len(index_y) > 1:
        #             index_minval = np.argmin(z_indices_lower[index_y])
        #             indices_to_delete = np.delete(index_y,index_minval)
        #             # for mm in indices_to_delete:
        #             #     Lower_half[pixel_val,z_indices_lower[mm]] = 0
        #             y_indices_lower = np.delete(y_indices_lower,indices_to_delete)
        #             z_indices_lower = np.delete(z_indices_lower,indices_to_delete)
            
        #     #Upper lower half
        #     Upper_half = V_temp[:,half_point:dim2]
        #     y_indices_upper, z_indices_upper = np.where(Upper_half == 1)
            

        #     for pixel_val in np.unique(y_indices_upper):
        #         index_y = np.where(y_indices_upper == pixel_val)[0]
        #         if len(index_y) > 1:
        #             index_maxval = np.argmax(z_indices_upper[index_y])
        #             indices_to_delete = np.delete(index_y,index_maxval)
        #             # for mm in indices_to_delete:
        #             #     Upper_half[pixel_val,z_indices_upper[mm]] = 0
        #             y_indices_upper = np.delete(y_indices_upper,indices_to_delete)
        #             z_indices_upper = np.delete(z_indices_upper,indices_to_delete)
                    
            
        #     #Adjust for cropping:
        #     z_indices_upper+=half_point
        #     y_indices_lower+=i
        #     y_indices_upper+=i
            
        #     #Check if they are empty
        #     if len(y_indices_lower) == 0 or len(y_indices_upper) == 0:
        #         i+=1
        #         continue
            
        #     distances_temp = {}
        #     for ii in range(len(y_indices_lower)):
        #         for jj in range(len(z_indices_upper)):
        #             x1, y1 = y_indices_lower[ii], z_indices_lower[ii]
        #             x2, y2 = y_indices_upper[jj], z_indices_upper[jj]
        #             # if 10 <= i <=(dim1-10):
        #             #     dist = y2-y1
        #             #     flag=1
        #             # else:
        #             # Calculate distance using Euclidean formula
        #             dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        #             flag=0
        #             distances_temp[(ii,jj,flag)] = dist
                    
    
        #     # Find the key with the lowest value
        #     min_value = min(distances_temp.values())
        #     min_key = next(key for key, value in distances_temp.items() if value == min_value)
        #     min_y_lower = y_indices_lower[min_key[0]]
        #     min_z_lower = z_indices_lower[min_key[0]]
        #     min_y_upper = y_indices_upper[min_key[1]]
        #     min_z_upper = z_indices_upper[min_key[1]]
            
        #     pixels_lower = (min_y_lower,min_z_lower,min_y_upper,min_z_upper)
            
        #     if flag == 1:
        #         min_y_lower = x1
        #         min_z_lower = y1
        #         min_y_upper = x2
        #         min_z_upper = y2
        #         min_value = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
        #     if min_value < 20:
        #         pass
        #     else:
        #         distances_min_list.append(min_value)
        #         pixels_min_list.append(pixels_lower)
        #         # show_one_slice_with_pixel_markings(data = V_body, pixels = pixels_lower, subject=subject, color = 'blue')
            
        #     max_value = max(distances_temp.values())
        #     max_key = next(key for key, value in distances_temp.items() if value == max_value)
        #     max_y_lower = y_indices_lower[max_key[0]]
        #     max_z_lower = z_indices_lower[max_key[0]]
        #     max_y_upper = y_indices_upper[max_key[1]]
        #     max_z_upper = z_indices_upper[max_key[1]]
            
        #     pixels_upper = (max_y_lower,max_z_lower,max_y_upper,max_z_upper)
            
        #     if flag == 1:
        #         max_y_lower = x1
        #         max_z_lower = y1
        #         max_y_upper = x2
        #         max_z_upper = y2
        #         max_value = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        #     if max_value < 20:
        #         pass
        #     else:
        #         distances_max_list.append(max_value)
        #         pixels_max_list.append(pixels_upper)
        #         # show_one_slice_with_pixel_markings(data = V_body, pixels = pixels_upper, subject=subject, color = 'red')
            
        #     # show_one_slice_with_pixel_markings(data = V_body, pixels = pixels_lower, subject=subject, color = 'blue')
        #     # show_one_slice_with_pixel_markings(data = V_body, pixels = pixels_upper, subject=subject, color = 'red')
            
        #     #Update counter
        #     i+=1
            
            
        # # Min pixels!
        # # for pixels in pixels_min_list:
        # #     show_one_slice_with_pixel_markings(data = V_body, pixels = pixels, subject=subject, color = 'blue')
           
        # # # break
        # #Max pixels!
        # # for pixels in pixels_max_list:
        # #     show_one_slice_with_pixel_markings(data = V_body, pixels = pixels, subject=subject, color = 'red')
                
        # def is_it_a_full_column(array):
        #     occurrences = 0 
        #     process_started = 0
        #     len_of_chain = 0
        #     for i in range(len(array)-1):
        #         if array[i] == 1:
                    
        #             process_started = 1
        #             len_of_chain+=1
                    
        #             if occurrences == 1: #Så betyder det, at den allerede har set den én gang.
        #                 return False, 0
        #         else:
        #             if process_started == 1:
        #                 occurrences=1
        #             else:
        #                 pass
        #     return True, len_of_chain
        
        # def find_longest_chain(array):
        #     unique_ID = 0 
        #     process_started = 0
        #     chain = []
        #     len_of_chain = []
        #     for i in range(len(array)):
        #         if array[i] == 1:
        #             chain.append(unique_ID)
        #         else:
        #             unique_ID+=1
        #     chain = np.array(chain)
        #     for unique_val in np.unique(chain):
        #         len_of_chain.append(sum(chain==unique_val))
        #     return max(len_of_chain)
        
        
        # #Remove last elements in list
        # stop=False
        # i=1
        # distances_min_list_original = copy.copy(distances_min_list)
        # pixels_min_list_original = copy.copy(pixels_min_list)
        # while not stop:
        #     pixel = pixels_min_list_original[-i]
        #     firstpixel = pixel[0]
        #     secondpixel = pixel[2]
            
        #     firstcolumn = V_body[firstpixel,:]
        #     secondcolumn = V_body[secondpixel,:]
            
            
        #     is_first_full,len_of_chain1 = is_it_a_full_column(firstcolumn)
        #     is_second_full,len_of_chain2 = is_it_a_full_column(secondcolumn)
            
            
        #     if is_first_full or is_second_full:
        #         if len_of_chain1 > 15 or len_of_chain2 > 15:
        #             stop = True
        #         else:
        #             stop = False
        #     else:
        #         stop = False
        #     #Pop elements
        #     distances_min_list.pop()
        #     pixels_min_list.pop()
        #     #Update counter
        #     i+=1
            
            
        #     # len_of_chain1 = find_longest_chain(firstcolumn)
        #     # len_of_chain2 = find_longest_chain(secondcolumn)
            
        #     # if len_of_chain1 > 15 or len_of_chain2 > 15:
        #     #     stop = True
        #     # else:
        #     #     stop = False
        #     # #Pop elements
        #     # distances_min_list.pop()
        #     # pixels_min_list.pop()
        #     # #Update counter
        #     # i+=1
            
            
        #     # is_first_full,len_of_chain1 = is_it_a_full_column(firstcolumn)
        #     # is_second_full,len_of_chain2 = is_it_a_full_column(secondcolumn)
        
        #     # if is_first_full or is_second_full:
        #     #     if len_of_chain1 > 15 or len_of_chain2 > 15:
        #     #         stop = True
        #     #     else:
        #     #         stop = False
        #     # else:
        #     #     stop = False
        #     # #Pop elements
        #     # distances_min_list.pop()
        #     # pixels_min_list.pop()
        #     # #Update counter
        #     # i+=1
            
            
        # #Check for too large jumps in the end
        # stop=False
        # i=1
        # # distances_min_list_original = copy.copy(distances_min_list)
        # # pixels_min_list_original = copy.copy(pixels_min_list)
        # while not stop:
        #     if abs(distances_min_list[-i]-distances_min_list[-(i+1)]) >= 3:
        #         stop = False
        #         #Pop elements
        #         distances_min_list.pop()
        #         pixels_min_list.pop()
        #         #Update counter
        #         i+=1
        #     else:
        #         stop = True
                
                
                
            
        # #Remove first elements in list
        # stop=False
        # i=0
        # distances_min_list_original2 = copy.copy(distances_min_list)
        # pixels_min_list_original2 = copy.copy(pixels_min_list)
        # while not stop:
        #     pixel = pixels_min_list_original2[i]
        #     firstpixel = pixel[0]
        #     secondpixel = pixel[2]
            
        #     firstcolumn = V_body[firstpixel,:]
        #     secondcolumn = V_body[secondpixel,:]
            
        #     len_of_chain1 = find_longest_chain(firstcolumn)
        #     len_of_chain2 = find_longest_chain(secondcolumn)
            
        #     if len_of_chain1 > 15 or len_of_chain2 > 15:
        #         stop = True
        #     else:
        #         stop = False
        #     #Pop elements
        #     distances_min_list.pop(0)
        #     pixels_min_list.pop(0)
        #     #Update counter
        #     i+=1
            
            
        #     # is_first_full,len_of_chain1 = is_it_a_full_column(firstcolumn)
        #     # is_second_full,len_of_chain2 = is_it_a_full_column(secondcolumn)
            
            
        #     # if is_first_full or is_second_full:
        #     #     if len_of_chain1 > 15 or len_of_chain2 > 15:
        #     #         stop = True
        #     #     else:
        #     #         stop = False
        #     # else:
        #     #     stop = False
        #     # #Pop elements
        #     # distances_min_list.pop(0)
        #     # pixels_min_list.pop(0)
        #     # #Update counter
        #     # i+=1
            
        # #Check for too large jumps in the beginning
        # stop=False
        # i=0
        # # distances_min_list_original = copy.copy(distances_min_list)
        # # pixels_min_list_original = copy.copy(pixels_min_list)
        # while not stop:
        #     if abs(distances_min_list[i]-distances_min_list[i+1]) >= 2:
        #         stop = False
        #         #Pop elements
        #         distances_min_list.pop(0)
        #         pixels_min_list.pop(0)
        #         #Update counter
        #         i+=1
        #     else:
        #         stop = True
                
                
        # #Largest distance must be in on of the two ends for the long part
        # filtering = np.ones(len(distances_max_list),dtype=bool)
        # filtering[6:-6] = False
        # distances_max_list = np.array(distances_max_list)[filtering]
        # pixels_max_list = np.array(pixels_max_list)[filtering]
                    
            
        # # # Min pixels!
        # # for pixels in pixels_min_list:
        # #     show_one_slice_with_pixel_markings(data = V_body, pixels = pixels, subject=subject, color = 'blue')
            
        # # # # # Max pixels!
        # # for pixels in pixels_max_list:
        # #     show_one_slice_with_pixel_markings(data = V_body, pixels = pixels, subject=subject, color = 'red')
                  
                

        # #Find out if there is a fracture
        # min_arg_dist = np.argmin(distances_min_list)
        # max_arg_dist = np.argmax(distances_max_list)
        
        # min_pixel = pixels_min_list[min_arg_dist]
        # max_pixel = pixels_max_list[max_arg_dist]
        
        # show_one_slice_with_pixel_markings(data = V_body, pixels = min_pixel, subject=subject, color = 'blue')
        # show_one_slice_with_pixel_markings(data = V_body, pixels = max_pixel, subject=subject, color = 'red')
        
        
        # min_dist = np.min(distances_min_list)
        # max_dist = np.max(distances_max_list)
        
        # reduction = ((max_dist-min_dist)/max_dist)*100
        
        # if reduction < 20:
        #     print("\nThere is no fracture in {}. Reduction is {}".format(my_v_dict[vertebra+1],np.round(reduction,1)))
        # elif 20 <= reduction < 25:
        #     print("\nFracture of grade 1 in {} detected! Reduction is {}".format(my_v_dict[vertebra+1],np.round(reduction,1)))
        # elif 25 <= reduction < 40:
        #     print("\nFracture of grade 2 in {} detected! Reduction is {}".format(my_v_dict[vertebra+1],np.round(reduction,1)))
        # elif reduction >= 40:
        #     print("\nFracture of grade 3 in {} detected! Reduction is {}".format(my_v_dict[vertebra+1],np.round(reduction,1)))
        # else:
        #     print("\nSomething went wrong with {}. Reduction is {}".format(my_v_dict[vertebra+1],np.round(reduction,1)))
            





























        # distances_min_list = np.delete(distances_min_list,indices_to_delete)
        # pixels_min_list = np.delete(pixels_min_list,indices_to_delete)
        # break
        
        # i=1
        # distances_min_list_original = distances_min_list
        # indices_to_delete = []
        # stop = False
        # while stop:
        #     if abs(distances_min_list[-(i)]-distances_min_list[-(i+1)]) > 0.5 and abs(distances_min_list[-(i)]-distances_min_list[-(i+1)]) > 0.5:
        #         indices_to_delete.append(-i)
        #     else:
        #         indices_to_delete.append(-(i+1))
        #         stop = True
        #     i+=1
                

                  
                
        # distances_min_list = np.delete(distances_min_list,indices_to_delete)
        # pixels_min_list = np.delete(pixels_min_list,indices_to_delete)
      


#SKRALD
                #     #Delete
                #     y_indices_upper = np.delete(y_indices_upper,indices_to_delete)
                #     z_indices_upper = np.delete(z_indices_upper,indices_to_delete)
                
                
                # for m in range(len(index_y)-1):
                    
            #         idx = index_y[m]
            #         if abs(z_indices_upper[idx] - z_indices_upper[idx+1]) == 1:
            #             Upper_half[pixel_val,z_indices_upper[idx]] = 0
                        
                # if len(index_y) > 2:
                #     index_maxval = np.argmax(z_indices_upper[index_y])
                #     indices_to_delete = np.delete(index_y,index_maxval)
                #     #Delete
                #     y_indices_upper = np.delete(y_indices_upper,indices_to_delete)
                #     z_indices_upper = np.delete(z_indices_upper,indices_to_delete)
                # else:
                #     pass
                
            # # # Append distance to the distances list
            # # distances.append(distance)
                
            #     dist = np.sqrt((y2 - y1) ** 2 + (z2 - z1) ** 2)
            
            # z_arg_min = np.argmin(z_indices)
            # z_arg_max = np.argmax(z_indices)
            
            # z1 = z_indices[z_arg_min] #minimum height
            # z2 = z_indices[z_arg_max] #maximum height
            # y1 = y_indices[z_arg_min]+i #y value at minimum height, PLUS i because of cropping
            # y2 = y_indices[z_arg_max]+i #y value at maximum height, PLUS i because of cropping
            
            
            
            # Upper_half = V_temp[half_point:dim1]
            
            
            
            
            # y_indices, z_indices = np.where(V_temp == 1)
            
            # z_arg_min = np.argmin(z_indices)
            # z_arg_max = np.argmax(z_indices)

            # z1 = z_indices[z_arg_min] #minimum height
            # z2 = z_indices[z_arg_max] #maximum height
            
            # y1 = y_indices[z_arg_min]+i #y value at minimum height, PLUS i because of cropping
            # y2 = y_indices[z_arg_max]+i #y value at maximum height, PLUS i because of cropping

            # dist = np.sqrt((y2 - y1) ** 2 + (z2 - z1) ** 2)
            
            # pixels = (y1,z1,y2,z2)
            # pixels_convert = (y1-i,z1,y2-i,z2)

            
            # if dist < 20: #Dont add if it is too small!
            #     show_one_slice_with_pixel_markings(data = V_temp, pixels = pixels_convert, subject=subject, color = 'green')
            #     pass
            # else:
            #     distances_list.append(dist)
            #     pixels_list.append(pixels)
            #     show_one_slice_with_pixel_markings(data = V_temp, pixels = pixels_convert, subject=subject, color = 'red')
            #     pass

            # # print("Distance between the two pixels:", distance)

            
            # for pixel_val in y_indices_lower:
            #     index_y = np.where(y_indices_lower == pixel_val)[0]
            #     if len(index_y) > 2:
            #         index_maxval = np.min(z_indices_lower[index_y])
            #         values_to_delete = np.delete(z_indices_lower[index_y],maxval)
            #     else:
            #         pass
            


# #Unpack coordinates
# x1,y1,x2,y2 = pixels

# data = copy.copy(V_temp)

# #Mark pixel as twos
# data[x1,y1] = 2
# data[x2,y2] = 2

# fig, ax = plt.subplots(figsize=(8, 6))
# # Define a custom colormap with black, white, and red
# colors = ['black', 'white', 'red']
# cmap = plt.cm.colors.ListedColormap(colors)

# # Plot the image using the custom colormap
# plt.imshow(data.T, cmap=cmap,origin="lower")

# # Draw a line between the two pixels
# plt.plot([x1,x2],[y1,y2], color='red', linewidth=1)
# # plt.xlim()

# # Set the colorbar to show the colormap values
# # plt.colorbar(ticks=[0, 1, 2], label='Color')

# # Show the plot
# plt.show()







        # while i+6 <= dim1:
        #     V_temp = V_body[i:i+6,:]
            
        #     # show_one_slice(V_temp,subject)
            
        #     y_indices, z_indices = np.where(V_temp == 1)
    
        # #Screening body
        # for i in range(dim1):
            
        #     V_temp = contour_image[i:i+1,:]
            
        #     show_one_slice(V_temp,subject)
            
        #     y_indices, z_indices = np.where(V_temp == 1)
            
        #     z_arg_min = np.argmin(z_indices)
            # z_arg_max = np.argmax(z_indices)

            # z1 = z_indices[z_arg_min]+i #minimum height, PLUS i because of cropping
            # z2 = z_indices[z_arg_max]+i #maximum height, PLUS i because of cropping
            
            # y1 = y_indices[z_arg_min]+i #y value at minimum height, PLUS i because of cropping
            # y2 = y_indices[z_arg_max]+i #y value at maximum height, PLUS i because of cropping

            # dist = np.sqrt((y2 - y1) ** 2 + (z2 - z1) ** 2)
            # dist = z2-z1
            
            
            # # pixels = (y1-i,z1-i,y2-i,z2-i)

            
            # if dist < 20: #Dont add if it is too small!
            #     show_one_slice_with_pixel_markings(data = V_temp, pixels = pixels, subject=subject, color = 'green')
            #     pass
            # else:
            #     distances.append(dist)
            #     show_one_slice_with_pixel_markings(data = V_temp, pixels = pixels, subject=subject, color = 'red')
    
            
        # #Find out it there is a fracture
        # min_dist = np.min(distances)
        # max_dist = np.max(distances)
        
        # reduction = ((max_dist-min_dist)/max_dist)*100
        
        # if reduction < 20:
        #     print("\nThere is no fracture in {}. Reduction is {}".format(my_v_dict[vertebra+1],np.round(reduction,1)))
        # elif 20 <= reduction < 25:
        #     print("\nFracture of grade 1 in {} detected! Reduction is {}".format(my_v_dict[vertebra+1],np.round(reduction,1)))
        # elif 25 <= reduction < 40:
        #     print("\nFracture of grade 2 in {} detected! Reduction is {}".format(my_v_dict[vertebra+1],np.round(reduction,1)))
        # elif reduction >= 40:
        #     print("\nFracture of grade 3 in {} detected! Reduction is {}".format(my_v_dict[vertebra+1],np.round(reduction,1)))
        # else:
        #     print("\nSomething went wrong with {}. Reduction is {}".format(my_v_dict[vertebra+1],np.round(reduction,1)))


        
































































#This code works!
# for subject in tqdm(all_subjects):
    
#     #Load centroids
#     ctd_list = np.load(os.path.join(dir_centroids,subject+'-centroids.npy'))
#     #Load mask
#     msk_nib = nib.load(os.path.join(dir_segmentations,subject+'-FULLSEGMENTATION.nii.gz'))
#     mask = np.asanyarray(msk_nib.dataobj, dtype=np.float32)    
    
#     # show_mask_dim1(mask,subject)
    
#     N_vertebrae = len(np.unique(mask))
        
#     #Isolate vertebrae
#     for vertebra in range(N_vertebrae-1): #Fordi vi vil ikke have 0 med, som er baggrund
    
#         #Find central slice!
#         centroid = ctd_list[vertebra]
        
#         x_coordinate = int(np.round(centroid[0]))
#         y_coordinate = int(np.round(centroid[1]))
#         z_coordinate = int(np.round(centroid[2]))
        
#         #Crop central slice
#         # V = V[x_min:x_max+1,y_coordinate,:]
        
#         V = mask[x_coordinate,:,:]
        
#         V = np.where(V == vertebra+1,1,0) #V for vertebrae
        
#         y_indices, z_indices = np.where(V != 0)
        
#         # Calculate the bounding box coordinates
#         y_min = np.min(y_indices)
#         y_max = np.max(y_indices)
#         z_min = np.min(z_indices)
#         z_max = np.max(z_indices)
        
#         V = V[y_min:y_max+1,z_min:z_max+1]
        
#         # show_one_slice(V,subject)
        
#         # for label in range(1,no_unique):
#         #     min_z_coordinates.append(np.min(np.where(blobs == label)[2]))
        
#         #Det her er default structure!
#         #kernel = np.array([[0, 1, 0],
#         #                   [1, 1, 1],
#         #                   [0, 1, 0]])
        
#         blobs, no_blobs = scipy.ndimage.label(V)
        
#         # show_one_slice(blobs,subject)
#         # show_one_slice(V,subject)

        
#         no_unique = len(np.unique(blobs)) #blobs er labelled array. no_blobs er antal blobs i den array.
        
#         max_y_coordinates = []
        
#         for label in range(1,no_unique):
#             max_y_coordinates.append(np.max(np.where(blobs == label)[0]))
        
    
#         Highest_blob_index = np.argmax(max_y_coordinates)+1 #Fordi jeg sortede 0 fra før oppe i forløbet. Se mit range. 0 er baggrunds-class
        
#         #Remove everything else than the body
#         V_body = np.where(blobs == Highest_blob_index,1,0)
        
#         # show_one_slice(V_body,subject)
        
#         #Crop again
#         y_indices, z_indices = np.where(V_body != 0)
        
#         # Calculate the bounding box coordinates
#         y_min = np.min(y_indices)
#         y_max = np.max(y_indices)
#         z_min = np.min(z_indices)
#         z_max = np.max(z_indices)
        
#         V_body = V_body[y_min:y_max+1,z_min:z_max+1]
#         # show_one_slice(V_body,subject)
        
#         dim1, _ = V_body.shape
       
#         distances = []
        
#         # if vertebra != 3:
#         #     continue
        
#         #Screening body
#         i=0
#         while i+6 <= dim1:
#             V_temp = V_body[i:i+6,:]
            
#             # show_one_slice(V_temp,subject)
            
#             y_indices, z_indices = np.where(V_temp == 1)
            
#             z_arg_min = np.argmin(z_indices)
#             z_arg_max = np.argmax(z_indices)

#             z1 = z_indices[z_arg_min]+i #minimum height, PLUS i because of cropping
#             z2 = z_indices[z_arg_max]+i #maximum height, PLUS i because of cropping
            
#             y1 = y_indices[z_arg_min]+i #y value at minimum height, PLUS i because of cropping
#             y2 = y_indices[z_arg_max]+i #y value at maximum height, PLUS i because of cropping

#             dist = np.sqrt((y2 - y1) ** 2 + (z2 - z1) ** 2)
            
#             pixels = (y1-i,z1-i,y2-i,z2-i)

            
#             if dist < 20: #Dont add if it is too small!
#                 # show_one_slice_with_pixel_markings(data = V_temp, pixels = pixels, subject=subject, color = 'green')
#                 pass
#             else:
#                 distances.append(dist)
#                 # show_one_slice_with_pixel_markings(data = V_temp, pixels = pixels, subject=subject, color = 'red')
    
#             # print("Distance between the two pixels:", distance)
#             i+=1
            
#         #Find out it there is a fracture
#         min_dist = np.min(distances)
#         max_dist = np.max(distances)
        
#         reduction = ((max_dist-min_dist)/max_dist)*100
        
#         if reduction < 20:
#             print("\nThere is no fracture in {}. Reduction is {}".format(my_v_dict[vertebra+1],np.round(reduction,1)))
#         elif 20 <= reduction < 25:
#             print("\nFracture of grade 1 in {} detected! Reduction is {}".format(my_v_dict[vertebra+1],np.round(reduction,1)))
#         elif 25 <= reduction < 40:
#             print("\nFracture of grade 2 in {} detected! Reduction is {}".format(my_v_dict[vertebra+1],np.round(reduction,1)))
#         elif reduction >= 40:
#             print("\nFracture of grade 3 in {} detected! Reduction is {}".format(my_v_dict[vertebra+1],np.round(reduction,1)))
#         else:
#             print("\nSomething went wrong with {}. Reduction is {}".format(my_v_dict[vertebra+1],np.round(reduction,1)))
        






























































            
        
        

        
        # #Set to zero!
        # # V = np.where(mask == i+1,1,0)
        
        # # show_mask_dim1(V,subject)
        
        # # Find the coordinates of non-zero (1) values
        # x_indices, y_indices, z_indices = np.where(V != 0) #
        
        # # Calculate the bounding box coordinates
        # x_min = np.min(x_indices)
        # x_max = np.max(x_indices)
        # y_min = np.min(y_indices)
        # y_max = np.max(y_indices)
        # z_min = np.min(z_indices)
        # z_max = np.max(z_indices)
        
        # # coordinates = (x_min,x_max,y_min,y_max,z_min,z_max)
        # # show_boundingbox_dim1(mask, coordinates, subject)
        
        # #Cut image from bounding box!
        # # V = V[x_min:x_max+1,y_min:y_max+1,z_min:z_max+1]
        
        # # show_mask_dim1(V,subject)
        
        # V = V[x_coordinate,:,z_min:z_max+1]

        

        
        # show_one_slice(V,subject)
        
        
        
        
        
        
        
        
        
        




    #Virkelig fed kode, der kan tegne bounding box!
    # for i in range(N_vertebrae-1): #Fordi vi vil ikke have 0 med, som er baggrund

    #     V = np.where(mask == i+1,1,0)
        
    #     # show_mask_dim1(V,subject)
        
    #     # Find the coordinates of non-zero (1) values
    #     x_indices, y_indices, z_indices = np.where(mask == i+1) #
        
    #     # Calculate the bounding box coordinates
    #     x_min = np.min(x_indices)
    #     x_max = np.max(x_indices)
    #     y_min = np.min(y_indices)
    #     y_max = np.max(y_indices)
    #     z_min = np.min(z_indices)
    #     z_max = np.max(z_indices)
        
    #     coordinates = (x_min,x_max,y_min,y_max,z_min,z_max)
        
    #     show_boundingbox_dim1(mask, coordinates, subject)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # mask = np.where(mask>0,1,0)
    
    # mask = remove_small_objects(mask)
    
    # msk_nib = nib.Nifti1Image(mask, msk_nib.affine) #img_nib.affine) #random_affine) #img_nib.affine) #DEFINE ANOTHER AFFINE!
    # nib.save(msk_nib, os.path.join(predictions_folder, subject+'-CLEANED.nii.gz'))
    
    
        
    # unique_labels = np.unique(mask)
    
    # # Step 2: Calculate label sizes
    # label_sizes = []
    # for label in unique_labels[1:]:  # Exclude background label 0
    #     label_sizes.append(np.sum(mask == label))
    
    # # Step 3: Find the index of the largest label
    # largest_label_index = np.argmax(label_sizes)
    
    # # Step 4: Create a new filtered mask
    # filtered_mask = np.zeros_like(mask)
    # filtered_mask[mask == unique_labels[largest_label_index]] = unique_labels[largest_label_index]
    
    # # Optional: Remove small noise blobs using scipy.ndimage.label
    # labeled_mask, num_labels = label(filtered_mask)
    # label_counts = np.bincount(labeled_mask.flatten())
    # largest_label = np.argmax(label_counts[1:]) + 1
    # filtered_mask = (labeled_mask == largest_label)


# Now 'filtered_mask' contains the largest connected component, and small noise blobs are removed
        
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
    #     if area < 200:
    #         blobs[blobs==label] = 0
                    
    # msk_nib = nib.Nifti1Image(mask, msk_nib.affine) #img_nib.affine) #random_affine) #img_nib.affine) #DEFINE ANOTHER AFFINE!
    # nib.save(msk_nib, os.path.join(predictions_folder, subject+'-CLEANED2.nii.gz'))
    
            