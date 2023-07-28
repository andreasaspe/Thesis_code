# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 22:44:21 2023

@author: PC
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
from copy import deepcopy


import sklearn

df_HU = pd.read_csv('E:/s174197/data_RH/Predictions/HU_pred.csv')

print(len(df_HU))
print(min(df_HU['Target L1 HU']))
print(max(df_HU['Target L1 HU']))

abs_diff = abs(df_HU['Target L1 HU']-df_HU['Prediction L1 HU'])

print('Average distance between ground truth and target: {:.2f}'.format(abs_diff.mean()))


df_HU = df_HU[df_HU['Target L1 HU'] < 100] #Filtered HU

print(df_HU)