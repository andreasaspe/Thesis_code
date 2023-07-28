import os
import shutil
from natsort import natsorted

def copy_files(source_folder, destination_folder, substrings_to_match):
    for root, dirs, files in os.walk(source_folder):
        # Get the corresponding subfolder path in the destination folder
        subfolder = os.path.relpath(root, source_folder)
        destination_subfolder = os.path.join(destination_folder, subfolder)

        # Create the subfolder in the destination folder if it doesn't exist
        os.makedirs(destination_subfolder, exist_ok=True)

        # Sort files using natural sorting (both alphabetic and numeric order)
        sorted_files = natsorted(files)

        # Copy files starting with any of the substrings from the current subfolder
        for file in sorted_files:
            for substring in substrings_to_match:
                if file.startswith(substring):
                    source_file = os.path.join(root, file)
                    destination_file = os.path.join(destination_subfolder, file)
                    shutil.copy2(source_file, destination_file)
                    break

# Usage example
source_folder = '/scratch/s174197/data/Verse20/VertebraeLocalisation2/Verse20_validation_prep_alldata' #'/scratch/s174197/data/Verse20/VertebraeSegmentation/Verse20_validation_prep' #Verse20_training_rescaled
destination_folder = '/scratch/s174197/data/Verse20/VertebraeLocalisation2/SUBSET_Verse20_validation_prep_alldata' #'/scratch/s174197/data/Verse20/VertebraeSegmentation/SUBSET_Verse20_validation_prep' #SUBSET_Verse20_training_rescaled
#substrings_to_match = ['sub-verse500','sub-verse506']  #TRAINING Specify the list of substrings to match here
substrings_to_match = ['sub-verse505','sub-verse511','sub-verse513','sub-verse522']  #VALIDATION Specify the list of substrings to match here 508?


copy_files(source_folder, destination_folder, substrings_to_match)






# def copy_files(source_folder, destination_folder, num_files=10):
#     for root, dirs, files in os.walk(source_folder):
#         # Get the corresponding subfolder path in the destination folder
#         subfolder = os.path.relpath(root, source_folder)
#         destination_subfolder = os.path.join(destination_folder, subfolder)

#         # Create the subfolder in the destination folder if it doesn't exist
#         os.makedirs(destination_subfolder, exist_ok=True)

#         # Sort files by their names using the default system sorting order
#         sorted_files = sorted(files)

#         # Copy the first 'num_files' files from the current subfolder
#         for file in sorted_files[:num_files]:
#             source_file = os.path.join(root, file)
#             destination_file = os.path.join(destination_subfolder, file)
#             shutil.copy2(source_file, destination_file)

# # Usage example
# source_folder = '/scratch/s174197/data/Verse20/VertebraeSegmentation/Verse20_training_prep'
# destination_folder = '/scratch/s174197/data/Verse20/VertebraeSegmentation/SUBSET_Verse20_training_prep'
# num_files_to_copy = 16

# copy_files(source_folder, destination_folder, num_files_to_copy)