# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 16:34:06 2023

@author: PC
"""

import pydicom
import numpy as np
import os
from pathlib import Path
from pydicom.fileset import FileSet
from my_plotting_functions import *



# dcm = pydicom.dcmread(r"E:/s174197/L1_HU_test/DICOMDIR")

path = r"E:/s174197/L1_HU_test/DICOMDIR"
ds = pydicom.dcmread(path)

# private_tag_data_1 = dcm[(0x7a37, 0xafbc)].value
# private_tag_data_2 = dcm[(0xbc95, 0xa540)].value

# pixel_data_1 = np.frombuffer(private_tag_data_1, dtype=np.uint16)
# pixel_data_2 = np.frombuffer(private_tag_data_2, dtype=np.uint16)

root_dir = Path(ds.filename).resolve().parent
print(f'Root directory: {root_dir}\n')


fs = FileSet(ds)
root_path = fs.path
# returns all contained values if IMAGE level entries
file_ids = fs.find_values("ReferencedFileID")
for file_id in file_ids:
    # file_id is a list, unpack it into the components using *
    dcm_path = os.path.join(root_path, *file_id)
    print(dcm_path) # here you can collect the paths or load the dataset
    
    dicom_file = pydicom.dcmread(dcm_path)
    data = dicom_file.pixel_array
    
    show_one_slice(data,'hej')
    
    
    


# # Iterate through the PATIENT records
# for patient in ds.patient_records:
#     print(
#         f"PATIENT: PatientID={patient.PatientID}, "
#         f"PatientName={patient.PatientName}"
#     )

#     # Find all the STUDY records for the patient
#     studies = [
#         ii for ii in patient.children if ii.DirectoryRecordType == "STUDY"
#     ]
#     for study in studies:
#         descr = study.StudyDescription or "(no value available)"
#         print(
#             f"{'  ' * 1}STUDY: StudyID={study.StudyID}, "
#             f"StudyDate={study.StudyDate}, StudyDescription={descr}"
#         )

#         # Find all the SERIES records in the study
#         all_series = [
#             ii for ii in study.children if ii.DirectoryRecordType == "SERIES"
#         ]
#         for series in all_series:
#             # Find all the IMAGE records in the series
#             images = [
#                 ii for ii in series.children
#                 if ii.DirectoryRecordType == "IMAGE"
#             ]
#             plural = ('', 's')[len(images) > 1]

#             descr = getattr(
#                 series, "SeriesDescription", "(no value available)"
#             )
#             print(
#                 f"{'  ' * 2}SERIES: SeriesNumber={series.SeriesNumber}, "
#                 f"Modality={series.Modality}, SeriesDescription={descr} - "
#                 f"{len(images)} SOP Instance{plural}"
#             )

#             # Get the absolute file path to each instance
#             #   Each IMAGE contains a relative file path to the root directory
#             elems = [ii["ReferencedFileID"] for ii in images]
#             # Make sure the relative file path is always a list of str
#             paths = [[ee.value] if ee.VM == 1 else ee.value for ee in elems]
#             paths = [Path(*p) for p in paths]

#             # List the instance file paths
#             for p in paths:
#                 print(f"{'  ' * 3}IMAGE: Path={os.fspath(p)}")

#                 # Optionally read the corresponding SOP Instance
#                 # instance = dcmread(Path(root_dir) / p)
#                 # print(instance.PatientName)

















#ELLER
# from pathlib import Path # pathlib for easy path handling
# import pydicom # pydicom to handle dicom files
# import matplotlib.pyplot as plt
# import numpy as np
# import dicom2nifti # to convert DICOM files to the NIftI format
# import nibabel as nib # nibabel to handle nifti files

# head_mri_dicom = Path("/kaggle/input/zenodo-mri-dicom-data-set/SE000001")
# path = r'E:/s174197/L1_HU_test/DICOMDIR'
# dicom2nifti.convert_directory(path, "r'E:/s174197/hvadsaa.nii.gz")

# # nifti = nib.load(hello)










#ELLER
# import os
# import pydicom
# import nibabel as nib

# def convert_dicom_to_nifti(dicomdir_path, output_dir):
#     # Load DICOMDIR file
#     dicomdir = pydicom.dcmread(dicomdir_path)

#     # Iterate over the studies in DICOMDIR
#     for study in dicomdir.DirectoryRecordSequence:
#         # Iterate over the series in each study
#         for series in study.ContainedSeries:
#             # Get the first DICOM file in the series
#             dicom_file = pydicom.dcmread(os.path.join(dicomdir_path, series.ReferencedSOPSequence[0].ReferencedSOPInstanceUID))
            
#             # Create the NIfTI image
#             nifti_img = nib.Nifti1Image(dicom_file.pixel_array, dicom_file.ImagePositionPatient)

#             # Save the NIfTI image
#             output_filename = f"{study.StudyInstanceUID}_{series.SeriesInstanceUID}.nii.gz"
#             output_path = os.path.join(output_dir, output_filename)
#             nib.save(nifti_img, output_path)

# # Example usage
# dicomdir_path = r'E:/s174197/L1_HU_test/DICOMDIR'
# output_dir = r'E:/s174197'

# convert_dicom_to_nifti(dicomdir_path, output_dir)