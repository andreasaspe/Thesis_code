#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 12:23:24 2023

@author: andreasaspe
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

data_type = 'test'
df_stage2 = pd.read_csv(os.path.join('/Users/andreasaspe/Documents/Data/Verse20/Predictions_dataframe',data_type,'df_stage2_elastic.csv'))
df_stage2 = df_stage2.sort_values(by='Average distance', ascending=True)

x = df_stage2[df_stage2.columns[2:]].values.flatten() #Flattened array
x = np.sort(x) #Sorted array
x = x[~np.isnan(x)] #Remove nans

print("REAL average HD: {}".format(np.mean(x)))
print("REAL median HD: {}".format(np.median(x)))


#Best and worse case
best_case = df_stage2.iloc[0].subjects
worst_case = df_stage2.iloc[-1].subjects

#Get median case
median_idx = int(np.round(len(df_stage2)/2))-1 #Fordi det er 0 indexeret!
#Extract median case
median_case = df_stage2.iloc[median_idx].subjects

print("The best case is {} which has an average distance of {:.2f}".format(best_case, df_stage2.loc[df_stage2['subjects'] == best_case, 'Average distance'].values[0]))
print("The median case is {} which has an average distance of {:.2f}".format(median_case, df_stage2.loc[df_stage2['subjects'] == median_case, 'Average distance'].values[0]))
print("The worst case is {} which has an average distance of {:.2f}".format(worst_case, df_stage2.loc[df_stage2['subjects'] == worst_case, 'Average distance'].values[0]))


# df_stage2['T10'].values(skipna=True)

boxprops = dict(linestyle='-', linewidth=2, color='k')
medianprops = dict(linestyle='-', linewidth=2, color='g')

# fig, ax = plt.subplots(ncols=1, figsize=(8,5))

plt.style.use('default')
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 15  # Default text - så fx. akse-tal osv.
plt.rcParams["axes.titlesize"] = 20  # Size for titles
plt.rcParams["axes.labelsize"] = 15  # Size for labels
plt.rcParams["legend.fontsize"] = 12  # Size for legends
plt.rcParams["figure.figsize"] = (12, 4) #(6.4, 4.8) #Fordi 15 er textwidth og jeg vil gerne have den 7 cm høj. Så skal være i forholdet 15/7

plt.figure()


# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "Helvetica"
# })

# plt.rcParams.update({
#     'text.usetex': True,
#     'font.family': 'sans-serif',
#     'font.sans-serif': ['Helvetica']
# })

#Fancy
boxplot = df_stage2.boxplot(column=['T10','T11','T12','L1','L2','L3','L4','L5'], grid=False, showfliers=True, showmeans=False,
                boxprops=boxprops, medianprops=medianprops,
                return_type='dict'
                )
#Not fancy
# boxplot = df_stage2.boxplot(column=['L5','L4','L3','L2','L1','T12','T11','T10'], grid=False, showfliers=True, showmeans=False,
#                 boxprops=boxprops, medianprops=medianprops,
#                 return_type='dict'
#                 )

# Get the current axes
ax = plt.gca()

# Set the font size of the numbers on the x-axis
ax.tick_params(axis='x')

# Set the font size of the numbers on the y-axis
ax.tick_params(axis='y')

plt.xlabel('Vertebra')
plt.ylabel('Displacement error [mm]')
plt.title('Vertebral Centroids Displacement Error')


# Display the plot
plt.show()



#STAGE 2 best
# list_of_subjects = ['sub-verse613','sub-verse809','sub-verse570','sub-verse752','sub-verse599','sub-verse760']
#STAGE 2 worst
# list_of_subjects = ['sub-verse502','sub-verse813','sub-verse649','sub-verse560','sub-verse626','sub-verse517']
