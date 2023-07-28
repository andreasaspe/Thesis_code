# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 16:12:44 2023

@author: PC
"""

import numpy as np

# Example arrays
array1 = np.array([[0.1, 0.6],
                   [0.3, 0.8]])

array2 = np.array([[0.4, 0.2],
                   [0.7, 0.9]])

array3 = np.array([[0.2, 0.3],
                   [0.6, 0.5]])

array4 = np.array([[0.5, 0.2],
                   [0.1, 0.4]])

array5 = np.array([[0.5, 0.1],
                   [0.9, 0.7]])

# List of arrays
arrays = [array1, array2, array3, array4, array5]

# Convert the list of arrays into a single NumPy array
arrays = np.array(arrays)

# Find the maximum values along the specified axis (axis=0 for elementwise comparison)
max_values = np.max(arrays, axis=0)

# Create a mask indicating where the maximum values are above 0.5
mask = max_values > 0.5

# Initialize an output array with zeros
output = np.zeros_like(max_values)

# Assign values based on the conditions
for i in range(len(arrays)):
    output[np.logical_and(mask, max_values == arrays[i])] = i + 1

# Print the resulting output array
print(output)