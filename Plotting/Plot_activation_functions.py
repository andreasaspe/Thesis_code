#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 15:27:57 2023

@author: andreasaspe
"""

#ReLU
import numpy as np
import matplotlib.pyplot as plt


# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the Leaky ReLU function
def leaky_relu(x, alpha=0.1):
    return np.maximum(alpha * x, x)

# Define the ReLU function
def relu(x):
    return np.maximum(0, x)

# Define the tanh function
def tanh(x):
    return np.tanh(x)

def plot_activation_function(activation='relu'):
    # Generate x values from -5 to 5
    x = np.linspace(-5, 5, 100)

    if activation == 'relu':
        # Compute ReLU values for the corresponding x values
        y = relu(x)

        # Set the title
        title = 'ReLU Function'
    elif activation == 'leaky_relu':
        # Compute Leaky ReLU values for the corresponding x values
        y = leaky_relu(x)

        # Set the title
        title = 'Leaky ReLU Function'
    elif activation == 'sigmoid':
        # Compute sigmoid values for the corresponding x values
        y = sigmoid(x)

        # Set the title
        title = 'Sigmoid Function'
    elif activation == 'tanh':
        # Compute tanh values for the corresponding x values
        y = tanh(x)

        # Set the title
        title = 'Hyperbolic Tangent (tanh) Function'
    else:
        print("Invalid activation function!")
        return

    # Create the figure and axis objects
    fig, ax = plt.subplots()

    # Plot the activation function
    ax.plot(x, y, color='blue', linewidth=2)

    # Set axis labels and title
    #ax.set_xlabel('x', fontsize=12)
    #ax.set_ylabel('y', fontsize=12)
    #ax.set_ylabel(activation.capitalize() + '(x)', fontsize=12)
    # ax.set_title("Leaky ReLU", fontsize=25)

    # Set the grid
    #ax.grid(True, linestyle='--', linewidth=0.5)

    # Remove the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

    # Show the plot
    plt.show()

# Call the plot_activation_function function with different activation functions
plot_activation_function('relu')
plot_activation_function('leaky_relu')
plot_activation_function('sigmoid')
plot_activation_function('tanh')