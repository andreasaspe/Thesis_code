# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 15:58:53 2023

@author: PC
"""

import pandas as pd
import pickle
import nibabel as nib
import numpy as np
import os
from my_plotting_functions import *


####ACTUALLY CREATE LIST ####
# df = pd.read_csv("G:\DTU-Vertebra-1\Metadata\DTU-Vertebrae-1_meta_data.csv")

# #Dont include ' Body 0.5 CE'. Det er contrast enhanced.
# condition1 = df[' SeriesDescription'] == ' Sft Tissue 0.5'
# condition2 = df[' SeriesDescription'] == ' Body 0.5'
# condition3 = df[' SeriesDescription'] == ' Body 0.5  CE'
# condition4 = df[' SeriesDescription'] == ' Body 0.5  Vol.'
# condition5 = df[' SeriesDescription'] == ' Body 0.5 CE'
# condition6 = df[' SeriesDescription'] == ' Body 0.5 CE Vol.'


# segmented_df = df[condition1 | condition2 | condition3 | condition4 | condition5 | condition6]

# series_list = segmented_df['pseudonymized_id'].to_list()

# with open("list_of_subjects", "wb") as fp:   #Pickling
#     pickle.dump(series_list, fp)
##############################

####### SEGMENTED LIST ########
df = pd.read_csv("G:\DTU-Vertebra-1\Metadata\DTU-Vertebrae-1_meta_data.csv")

#Dont include ' Body 0.5 CE'. Det er contrast enhanced.
condition1 = df[' SeriesDescription'] == ' Sft Tissue 0.5'
condition2 = df[' SeriesDescription'] == ' Body 0.5'
condition3 = df[' SeriesDescription'] == ' Body 0.5  CE'
condition4 = df[' SeriesDescription'] == ' Body 0.5  Vol.'
condition5 = df[' SeriesDescription'] == ' Body 0.5 CE'
condition6 = df[' SeriesDescription'] == ' Body 0.5 CE Vol.'
condition7 = df[' patient_id'].str.contains('FRACTURE', case=False)
condition8 = df[' patient_id'].str.contains('LOWHU', case=False)
condition9 = df[' patient_id'].str.contains('HEALTHY', case=False)


# segmented_df = df[condition5 & condition7]

#LOWHU
# segmented_df = df[condition5 & condition8]
# series_list = segmented_df['pseudonymized_id'].to_list() #pseudonymized_id,  patient_id
# series_list = np.array(series_list)
# idx_to_delete = np.where(series_list == 'VERTEBRAE_LOWHU_0115_SERIES0010')[0][0]
# series_list = list(np.delete(series_list,idx_to_delete))
# with open("list_of_subjects_LOWHU", "wb") as fp:   #Pickling
#     pickle.dump(series_list, fp)

#HEALTHY
# segmented_df = df[condition5 & condition9]
# series_list = segmented_df['pseudonymized_id'].to_list() #pseudonymized_id,  patient_id
# with open("list_of_subjects_HEALTHY", "wb") as fp:   #Pickling
#     pickle.dump(series_list, fp)

#FRACTURE
segmented_df = df[condition5 & condition7]
series_list = segmented_df['pseudonymized_id'].to_list() #pseudonymized_id,  patient_id
series_list = np.array(series_list)
idx_to_delete = np.where(series_list == 'VERTEBRAE_FRACTURE_0224_SERIES0000')[0][0]
series_list = list(np.delete(series_list,idx_to_delete))
series_list = np.array(series_list)
idx_to_delete = np.where(series_list == 'VERTEBRAE_FRACTURE_0255_SERIES0012')[0][0]
series_list = list(np.delete(series_list,idx_to_delete))
series_list = np.array(series_list)
idx_to_delete = np.where(series_list == 'VERTEBRAE_FRACTURE_0326_SERIES0017')[0][0]
series_list = list(np.delete(series_list,idx_to_delete))
with open("list_of_subjects_FRACTURE", "wb") as fp:   #Pickling
    pickle.dump(series_list, fp)
##############################



###### PLAYAROUND ######
# df = pd.read_csv("G:\DTU-Vertebra-1\Metadata\DTU-Vertebrae-1_meta_data.csv")

# #Dont include ' Body 0.5 CE'. Det er contrast enhanced.
# condition1 = df[' SeriesDescription'] == ' Sft Tissue 0.5'
# condition2 = df[' SeriesDescription'] == ' Body 0.5'
# condition3 = df[' SeriesDescription'] == ' Body 0.5  CE'

# segmented_df = df[condition1 | condition2 | condition3]

# series_list = segmented_df[['pseudonymized_id', ' patient_id',' SeriesDescription']]# .to_list()
        
# #series_list.to_csv('list_of_subjects.csv', index=False)

# dir_data = r'G:\DTU-Vertebra-1\NIFTI' #'/zhome/bb/f/127616/Documents/Thesis/Rawdata_training'

# series_list = series_list.to_numpy()

# for ID, patient, series_type in series_list:
#     if ID == 'VERTEBRAE_FRACTURE_0264_SERIES0039':
#         print(ID, patient, series_type)
        
#         filename_img = ID + ".nii.gz"
#         img_nib = nib.load(os.path.join(dir_data,filename_img))
#         img_data = np.asanyarray(img_nib.dataobj, dtype=np.float32)
        
#         show_slices_dim1(img_data,patient,no_slices=30)
##########################














   
#Old code: Det var da jeg først kun segmenterede ud fra original. Og ikke SeriesDescription
# df = pd.read_csv("G:\DTU-Vertebra-1\Metadata\DTU-Vertebrae-1_meta_data.csv")

# series_list = []

# for i in range(len(df[' ImageType'])):
#     if df[' ImageType'].iloc[i].find('ORIGINAL') != -1:
#         series_list.append(df['pseudonymized_id'].iloc[i])
        
# with open("list_of_subjects", "wb") as fp:   #Pickling
#    pickle.dump(series_list, fp)