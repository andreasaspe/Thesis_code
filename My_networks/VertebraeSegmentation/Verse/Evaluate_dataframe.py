#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 18:53:59 2023

@author: andreasaspe
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

data_type = 'test'
# which_checkpoint = 'normalrotation'
# which_checkpoint = 'betterrotation'
# which_checkpoint = 'onlyelastic'
which_checkpoint = 'evenbetterrotation'
df_DSC = pd.read_csv(os.path.join('/Users/andreasaspe/Documents/Data/Verse20/Predictions_dataframes_from_titans',data_type,which_checkpoint,'df_DSC.csv'))
df_HD = pd.read_csv(os.path.join('/Users/andreasaspe/Documents/Data/Verse20/Predictions_dataframes_from_titans',data_type,which_checkpoint,'df_HAUSDORFF.csv'))

#Remove wierd L2:
row_index = df_DSC[df_DSC['subjects'] == 'sub-gl279'].index
df_DSC.loc[row_index, 'L2'] = np.nan
df_HD.loc[row_index, 'L2'] = np.nan

#############################################################
######################### DICE SCORE ########################
#############################################################
df_DSC = df_DSC.sort_values(by='Average distance', ascending=False) #False fordi højst er bedst

#Best and worse case
best_case = df_DSC.iloc[0].subjects
worst_case = df_DSC.iloc[-1].subjects

#Get median case
median_idx = int(np.round(len(df_DSC)/2))-1 #Fordi det er 0 indexeret!
#Extract median case
median_case = df_DSC.iloc[median_idx].subjects

print("DICE SCORE")
print("The best case is {} which has an average distance of {:.2f}".format(best_case, df_DSC.loc[df_DSC['subjects'] == best_case, 'Average distance'].values[0]))
print("The median case is {} which has an average distance of {:.2f}".format(median_case, df_DSC.loc[df_DSC['subjects'] == median_case, 'Average distance'].values[0]))
print("The worst case is {} which has an average distance of {:.2f}".format(worst_case, df_DSC.loc[df_DSC['subjects'] == worst_case, 'Average distance'].values[0]))
print("Mean of all dice scores: {:.2f}".format(np.mean(df_DSC['Average distance'].values)))

# print()

vertebrae_columns = ['L5', 'L4', 'L3', 'L2', 'L1', 'T12',
       'T11', 'T10']

x = df_DSC[vertebrae_columns].values.flatten() #Flattened array
x = np.sort(x) #Sorted array
x = x[~np.isnan(x)] #Remove nans

print("REAL average DSC: {}".format(np.mean(x)))
print("REAL median DSC: {}".format(np.median(x)))


#Get values value
min_val = np.min(x)
max_val = np.max(x)
median_idx = int(np.round(len(x)/2))-1 #Works because it is sorted
median_val = x[median_idx]

print("DICE")
#min val
bool_df = df_DSC.eq(min_val)
subject_min = df_DSC.loc[bool_df.any(axis=1),'subjects'].values[0]
vertebra_min = bool_df.any(axis=0).index[bool_df.any(axis=0)][0]
print()
print("Min:")
print(subject_min,vertebra_min,min_val)
#median val
bool_df = df_DSC.eq(median_val)
subject_median = df_DSC.loc[bool_df.any(axis=1),'subjects'].values[0]
vertebra_median = bool_df.any(axis=0).index[bool_df.any(axis=0)][0]
print()
print("Median:")
print(subject_median,vertebra_median,median_val)
#max val
bool_df = df_DSC.eq(max_val)
subject_max = df_DSC.loc[bool_df.any(axis=1),'subjects'].values[0]
vertebra_max = bool_df.any(axis=0).index[bool_df.any(axis=0)][0]
print()
print("Max:")
print(subject_max,vertebra_max,max_val)

L5_mean  = df_DSC['L5'].mean()
L4_mean  = df_DSC['L4'].mean()
L3_mean  = df_DSC['L3'].mean()
L2_mean  = df_DSC['L2'].mean()
L1_mean  = df_DSC['L1'].mean()
T12_mean  = df_DSC['T12'].mean()
T11_mean  = df_DSC['T11'].mean()
T10_mean  = df_DSC['T10'].mean()

print('L5',np.round(L5_mean,3))
print('L4',np.round(L4_mean,3))
print('L3',np.round(L3_mean,3))
print('L2',np.round(L2_mean,3))
print('L1',np.round(L1_mean,3))
print('T12',np.round(T12_mean,3))
print('T11',np.round(T11_mean,3))
print('T10',np.round(T10_mean,3))
print('Average',np.round(np.mean([L5_mean,L4_mean,L3_mean,L2_mean,L1_mean,T12_mean,T11_mean,T10_mean]),3))


plt.style.use('default')
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 15  # Default text - så fx. akse-tal osv.
plt.rcParams["axes.titlesize"] = 20  # Size for titles
plt.rcParams["axes.labelsize"] = 15  # Size for labels
plt.rcParams["legend.fontsize"] = 12  # Size for legends
plt.rcParams["figure.figsize"] = (12, 4) #(6.4, 4.8) #Fordi 15 er textwidth og jeg vil gerne have den 7 cm høj. Så skal være i forholdet 15/7

plt.figure()

boxprops = dict(linestyle='-', linewidth=2, color='k')
medianprops = dict(linestyle='-', linewidth=2, color='g')

#Fancy
boxplot = df_DSC.boxplot(column=['T10','T11','T12','L1','L2','L3','L4','L5'], grid=False, showfliers=True, showmeans=False,
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
plt.ylabel('DSC')
plt.title('Dice Score')
# Display the plot
plt.show()


#############################################################
##################### HAUSDORFF DISTANCE ####################
#############################################################
df_HD = df_HD.sort_values(by='Average distance', ascending=True) #True fordi lavest er bedst

#Best and worse case
best_case = df_HD.iloc[0].subjects
worst_case = df_HD.iloc[-1].subjects

#Get median case
median_idx = int(np.round(len(df_HD)/2))-1 #Fordi det er 0 indexeret!
#Extract median case
median_case = df_HD.iloc[median_idx].subjects

print("HAUSDORFF DISTANCE")
print("The best case is {} which has an average distance of {:.2f}".format(best_case, df_HD.loc[df_HD['subjects'] == best_case, 'Average distance'].values[0]))
print("The median case is {} which has an average distance of {:.2f}".format(median_case, df_HD.loc[df_HD['subjects'] == median_case, 'Average distance'].values[0]))
print("The worst case is {} which has an average distance of {:.2f}".format(worst_case, df_HD.loc[df_HD['subjects'] == worst_case, 'Average distance'].values[0]))
print("Mean of all HD: {:.2f}".format(np.mean(df_HD['Average distance'].values)))
print()


vertebrae_columns = ['L5', 'L4', 'L3', 'L2', 'L1', 'T12',
       'T11', 'T10']

x = df_HD[vertebrae_columns].values.flatten() #Flattened array
x = np.sort(x) #Sorted array
x = x[~np.isnan(x)] #Remove nans

print("REAL average HD: {}".format(np.mean(x)))
print("REAL median HD: {}".format(np.median(x)))

#Get values value
min_val = np.min(x)
max_val = np.max(x)
median_idx = int(np.round(len(x)/2))-1 #Works because it is sorted
median_val = x[median_idx]

print()
print("HAUSDORFF")
#min val
bool_df = df_HD.eq(min_val)
subject_min = df_DSC.loc[bool_df.any(axis=1),'subjects'].values[0]
vertebra_min = bool_df.any(axis=0).index[bool_df.any(axis=0)][0]
print()
print("Min:")
print(subject_min,vertebra_min,min_val)
#median val
bool_df = df_HD.eq(median_val)
subject_median = df_DSC.loc[bool_df.any(axis=1),'subjects'].values[0]
vertebra_median = bool_df.any(axis=0).index[bool_df.any(axis=0)][0]
print()
print("Median:")
print(subject_median,vertebra_median,median_val)
#max val
bool_df = df_HD.eq(max_val)
subject_max = df_DSC.loc[bool_df.any(axis=1),'subjects'].values[0]
vertebra_max = bool_df.any(axis=0).index[bool_df.any(axis=0)][0]
print()
print("Max:")
print(subject_max,vertebra_max,max_val)


L5_mean  = df_HD['L5'].mean()
L4_mean  = df_HD['L4'].mean()
L3_mean  = df_HD['L3'].mean()
L2_mean  = df_HD['L2'].mean()
L1_mean  = df_HD['L1'].mean()
T12_mean  = df_HD['T12'].mean()
T11_mean  = df_HD['T11'].mean()
T10_mean  = df_HD['T10'].mean()

print('L5',np.round(L5_mean,2))
print('L4',np.round(L4_mean,2))
print('L3',np.round(L3_mean,2))
print('L2',np.round(L2_mean,2))
print('L1',np.round(L1_mean,2))
print('T12',np.round(T12_mean,2))
print('T11',np.round(T11_mean,2))
print('T10',np.round(T10_mean,2))
print('Average',np.round(np.mean([L5_mean,L4_mean,L3_mean,L2_mean,L1_mean,T12_mean,T11_mean,T10_mean]),2))


plt.figure()

boxprops = dict(linestyle='-', linewidth=2, color='k')
medianprops = dict(linestyle='-', linewidth=2, color='g')

#Fancy
boxplot = df_HD.boxplot(column=['T10','T11','T12','L1','L2','L3','L4','L5'], grid=False, showfliers=True, showmeans=False,
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
plt.ylabel('HD [mm]')
plt.title('Hausdorff Distance')
# Display the plot
plt.show()