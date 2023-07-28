#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 11:43:05 2023

@author: andreasaspe
"""

from my_data_utils import *
from my_plotting_functions import *
import math
import nibabel as nib


import numpy as np

def new_gauss(origins, meshgrid_dim, gamma, sigma=1):
    d=3 #dimension
    x,y,z = origins
    mesh_x,mesh_y,mesh_z = meshgrid_dim
    
    x_grid, y_grid, z_grid = np.meshgrid(np.arange(mesh_x), np.arange(mesh_y),np.arange(mesh_z),indexing='ij')
    kernel = np.exp(-((x_grid - x)**2 + (y_grid - y)**2 + (z_grid - z)**2) / (2 * sigma**2))
    factor = gamma/( (2*math.pi)**(d/2)*sigma**d   )
    heatmap = factor*kernel
    return heatmap

gamma = 1
origins = (50,50,50)
meshgrid_dim = (101,101,101)
sigma = 15

heatmap = new_gauss(origins,meshgrid_dim,gamma, sigma)

heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

show_heatmap_dim1(heatmap,subject='dontknow',no_slices=20)




# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# data = heatmap[:15,:,:]

# # Generate sample data
# x = np.linspace(0, 1, 15)
# y = np.linspace(0, 1, 30)
# z = np.linspace(0, 1, 30)
# X, Y, Z = np.meshgrid(x, y, z)

# # Flatten the arrays
# X = X.flatten()
# Y = Y.flatten()
# Z = Z.flatten()
# values = data.flatten()


# # Create a 3D scatter plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot the scatter points with colors based on values
# scatter = ax.scatter(X,Y,Z, c=values, cmap='hot',alpha=1.0*(data.T > 0.3),s=100)

# # Add a colorbar
# fig.colorbar(scatter)

# # Set labels and title
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('3D Scatter Plot with Color')

# # Show the plot
# plt.show()
