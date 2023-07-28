#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 10:18:41 2023

@author: andreasaspe
"""

#General imports
import os
import sys
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.transforms as transforms
from torch import linalg as LA
import wandb
import SimpleITK as sitk
import elasticdeform
import nibabel as nib
#My own documents
from my_plotting_functions import *
from Create_dataset import LoadData
from my_data_utils import Predict, gaussian_kernel_3d
#from VertebraeLocalisationNet import *
# from new_VertebraeLocalisationNet import *
#from VertebraeLocalisationNet_newdropout import *

#Define paramters
parameters_dict = {
    'epochs': 3000,
    'learning_rate': 1e-5, #1e-5, # 1e-8
    'weight_decay': 0.0001,
    'batch_size': 1,
    'dropout': 0.0
}


def elastically_deform_image(sitk_image, dfield_image, num_control_points, std_dev):
    """Implements elastic deformations that are used for augmentation, using the pre-made SimpleITK library.
    The number of control points is (size of the grid - 2) due to borders, while std_dev is the standard
    deviation of the displacement vectors in pixels. Interpolator is either 'linear' or 'cubic'"""

    # Allocate memory for transform parameters
    transform_mesh_size = [num_control_points] * sitk_image.GetDimension()
    transform = sitk.BSplineTransformInitializer(
        sitk_image ,
        transform_mesh_size
    )

    # Read the parameters as a numpy array, then add random
    # displacement and set the parameters back into the transform
    params = np.asarray(transform.GetParameters(), dtype=np.float64)
    params = params + np.random.randn(params.shape[0]) * std_dev
    transform.SetParameters(tuple(params))

    # Create resampler object
    # The interpolator can be set to sitk.sitkBSpline for cubic interpolation,
    # see https://itk.org/SimpleITKDoxygen/html/namespaceitk_1_1simple.html#a7cb1ef8bd02c669c02ea2f9f5aa374e5 for more options
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitk_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-2048)  # todo: get defalt CT value from config
    resampler.SetTransform(transform)

    # Execute augmentation
    sitk_augmented_image = resampler.Execute(sitk_image)

    resampler.SetDefaultPixelValue(5)  # todo: get max distance value from config
    dfield_augmented_image = resampler.Execute(dfield_image)

    return sitk_augmented_image, dfield_augmented_image


# image = sitk.ReadImage('/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation2/Verse20_training_prep/img/sub-verse500_img.nii.gz')
# image_array = sitk.GetArrayViewFromImage(image)
# # show_slices_dim1(image_array,'hej')


# sitk_augmented_image, dfield_augmented_image = elastically_deform_image(image,image,10,10)
# image_array = sitk.GetArrayViewFromImage(sitk_augmented_image)

# show_slices_dim1(image_array,'hej')


# image = nib.load('/Users/andreasaspe/Documents/Data/Verse20/VertebraeLocalisation2/Verse20_training_prep/img/sub-verse500_img.nii.gz')
# image = image.get_fdata()
img = np.load('/Users/andreasaspe/Documents/Data/Verse20/VertebraeSegmentation/Verse20_training_prep/img/sub-verse507-18_img.npy')
msk = np.load('/Users/andreasaspe/Documents/Data/Verse20/VertebraeSegmentation/Verse20_training_prep/msk/sub-verse507-18_msk.npy')

# show_mask_img_dim1(img, msk, 'hej')

# apply deformation with a random 3 x 3 grid
[img_deformed, msk_deformed] = elasticdeform.deform_random_grid([img, msk], sigma=2, points=6, cval=-1, order=[3, 0])
#sigma = 2, point = 10

show_mask_img_dim1(img_deformed, msk_deformed, 'hej')

# image_deformed = elasticdeform.deform_random_grid(image, sigma=2, points=10,cval = -1)

# show_slices_dim1(image_deformed,'hej')
# show_slices_dim2(image_deformed,'hej')
# show_slices_dim3(image_deformed,'hej')
