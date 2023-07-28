#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 21:30:58 2023

@author: andreasaspe
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

df_stage1 = pd.read_csv('/Users/andreasaspe/Documents/Data/Verse20/Predictions/df_stage1.csv')
df_stage1 = df_stage1.sort_values(by='iou', ascending=False) #Hvis det var mse skulle det hedde true

#Best and worse case
best_case = df_stage1.iloc[0].subjects
worst_case = df_stage1.iloc[-1].subjects

#Get median case
median_idx = int(np.round(len(df_stage1)/2))-1 #Fordi det er 0 indexeret!
#Extract median case
median_case = df_stage1.iloc[median_idx].subjects

print("The best case is {} which has a iou of {:.2f}".format(best_case, df_stage1.loc[df_stage1['subjects'] == best_case, 'iou'].values[0]))
print("The median case is {} which has a iou of {:.2f}".format(median_case, df_stage1.loc[df_stage1['subjects'] == median_case, 'iou'].values[0]))
print("The worst case is {} which has a iou of {:.2f}".format(worst_case,  df_stage1.loc[df_stage1['subjects'] == worst_case, 'iou'].values[0]))


# print("Average iou is {}".format(np.mean(df_stage1['iou'].values)))
# print("Minimum iou is {}".format(np.min(df_stage1['iou'].values)))
# print("Maximum iou is {}".format(np.max(df_stage1['iou'].values)))