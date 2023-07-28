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

df_pred = pd.read_csv('E:/s174197/data_RH/Predictions/Fracture_pred.csv')
df_GT = pd.read_csv('E:/s174197/data_RH/Predictions/Fracture_GT.csv')


# Function to map values to 1 or 0
def map_to_binary1(value):
    return 1 if value >= 1 else 0

def map_to_binary2(value):
    return 1 if value >= 2 else 0

def map_to_binary3(value):
    return 1 if value >= 3 else 0

# # Apply the function to all columns except the first one
# df_pred[df_pred.columns[1:]] = df_pred[df_pred.columns[1:]].apply(lambda x: x.apply(map_to_binary2))
# df_GT[df_GT.columns[1:]] = df_GT[df_GT.columns[1:]].apply(lambda x: x.apply(map_to_binary2))

# hej = []
length = 0
for i in range(8):
    df_pred_temp = deepcopy(df_pred)
    df_GT_temp = deepcopy(df_GT)
    col_name = df_pred_temp.columns[i+1]
    df_pred_temp = df_pred_temp[col_name]
    df_GT_temp = df_GT_temp[col_name]
    
    pred_numpy = df_pred_temp.values.flatten()
    GT_numpy = df_GT_temp.values.flatten()

    idx_to_delete = np.where(pred_numpy == -1)[0]

    pred_numpy = np.delete(pred_numpy,idx_to_delete)
    GT_numpy = np.delete(GT_numpy,idx_to_delete)
    
    length+=len(GT_numpy)

    pred_numpy = np.where(pred_numpy>=1,1,0)
    GT_numpy = np.where(GT_numpy>=1,1,0)
    
    cm = confusion_matrix(GT_numpy, pred_numpy)
    
    TN = cm[0,0]
    FN = cm[1,0]
    TP = cm[1,1]
    FP = cm[0,1]
    
    acc = sklearn.metrics.accuracy_score(GT_numpy,pred_numpy)
    sens = TP/(TP+FN) #Same as recall and TPR (True Positive Rate)
    spec = TN/(TN+FP)
    precision = TP/(TP+FP)
    FPR = FP/(FP+TN) #False positive rate
    
    
    # print("{}: Accuracy is: {}".format(col_name,np.round(acc,2)))
    # print("{}: Sensitivity is: {}".format(col_name,np.round(sens,2))) #TPR
    # print("{}: Specificity is: {}".format(col_name,np.round(spec,2)))
    print("{}: Precision is: {}".format(col_name,np.round(precision,2)))
    # print("{}: FPR is: {}".format(col_name,np.round(FPR,2)))
# # #     hej.append(len(GT_numpy))
# # #     print(sum(GT_numpy))
    
# print(sum(hej))
pred_numpy = df_pred[df_pred.columns[1:]].values.flatten()
GT_numpy = df_GT[df_GT.columns[1:]].values.flatten()

idx_to_delete = np.where(pred_numpy == -1)[0]

pred_numpy = np.delete(pred_numpy,idx_to_delete)
GT_numpy = np.delete(GT_numpy,idx_to_delete)

# pred_numpy = np.where(pred_numpy>=1,1,0)
# GT_numpy = np.where(GT_numpy>=1,1,0)



#RANDOM
import random

# def generate_random_array(size):
#     num_zeros = size * 4 // 5
#     num_ones = size // 5

#     array = [0] * num_zeros + [1] * num_ones
#     random.shuffle(array)
#     return array

# def generate_random_array2(size):
#     num_zeros = size * 3 // 5
#     num_ones = size * 2// 5

#     array = [0] * num_zeros + [1] * num_ones
#     random.shuffle(array)
#     return array

def generate_random_array(size, probability_of_one):
    num_ones = int(size * probability_of_one)
    array = [0] * (size - num_ones) + [1] * num_ones
    random.shuffle(array)
    return array

# Generate a random array with 0.44% being 1s
array = generate_random_array(1000, 0.0044)  # Replace 1000 with the desired size of the array



acc_list = []
sens_list = []
spec_list = []
precision_list = []
FPR_list = []
 
# for i in range(1000):
#     # Generate two random arrays
#     GT_numpy = generate_random_array(690,1-530/690)  # Replace 10 with the desired size of the array
#     pred_numpy = generate_random_array(690,303/690)  # Replace 10 with the desired size of the array
    
#     # print("Array 1:", GT_numpy)
#     # print("Array 2:", generate_random_array2)
    
#     cm = confusion_matrix(GT_numpy, pred_numpy)
    
#     TN = cm[0,0]
#     FN = cm[1,0]
#     TP = cm[1,1]
#     FP = cm[0,1]
    
    
#     acc = sklearn.metrics.accuracy_score(GT_numpy,pred_numpy)
#     sens = TP/(TP+FN) #Same as recall
#     spec = TN/(TN+FP)
#     precision = TP/(TP+FP)
#     FPR = FP/(FP+TN) #False positive rate
    
#     acc_list.append(acc)
#     sens_list.append(sens)
#     spec_list.append(spec)
#     precision_list.append(precision)
#     FPR_list.append(FPR)
    
#     print("Accuracy is: {}".format(np.round(acc,2)))
#     print("Sensitivity is: {}".format(np.round(sens,2)))
#     print("Specificity is: {}".format(np.round(spec,2)))
#     print("Precision is: {}".format(np.round(precision,2)))
#     print("FPR is: {}".format(np.round(FPR,2)))

# print(np.mean(acc_list))
# print(np.mean(sens_list))
# print(np.mean(spec_list))
# print(np.mean(precision_list))
# print(np.mean(FPR_list))


#NOT RANDOM
# Calculate the confusion matrix
cm = confusion_matrix(GT_numpy, pred_numpy)

TN = cm[0,0]
FN = cm[1,0]
TP = cm[1,1]
FP = cm[0,1]

acc = sklearn.metrics.accuracy_score(GT_numpy,pred_numpy)
sens = TP/(TP+FN) #Same as recall
spec = TN/(TN+FP)
precision = TP/(TP+FP)
FPR = FP/(FP+TN) #False positive rate

print("Accuracy is: {}".format(np.round(acc,2)))
print("Sensitivity is: {}".format(np.round(sens,2)))
print("Specificity is: {}".format(np.round(spec,2)))
print("Precision is: {}".format(np.round(precision,2)))
print("FPR is: {}".format(np.round(FPR,2)))




# Create a DataFrame from the confusion matrix
class_names = ['No fracture', 'Grade 1', 'Grade 2', 'Grade 3']  # Replace with your class names
# class_names = ['No fracture', 'Fracture']  # Replace with your class names
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)

# Plot the confusion matrix using a heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='d', annot_kws={"size": 16})
# plt.title('Confusion Matrix')
plt.xlabel('Prediction')
plt.ylabel('GT')
plt.show()














# fpr = np.array([0,0.38,0.32,0.26,0.38,0.29,0.28,0.68,1])
# tpr = np.array([0,0.6,0.66,0.62,0.75,0.8,0.75,0.86,1])

# sort_indices = fpr.argsort()

# fpr = fpr[sort_indices]
# tpr = tpr[sort_indices]


# # cm = confusion_matrix(GT_numpy,pred_numpy)

# # fpr, tpr, thresholds = sklearn.metrics.roc_curve(GT_numpy, pred_numpy)

# # # Calculate the Area Under the Curve (AUC)
# roc_auc = auc(fpr, tpr)

# # # Plot the ROC curve
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc='lower right')
# plt.show()