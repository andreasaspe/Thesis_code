# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 19:08:02 2023

@author: PC
"""

import numpy as np

array = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])

def is_it_a_full_column(array):
    occurrences = 0 
    process_started = 0
    len_of_chain = 0
    for i in range(len(array)):
        if array[i] == 1:
            
            process_started = 1
            len_of_chain+=1
            
            if occurrences == 1: #Så betyder det, at den allerede har set den én gang.
                return False, 0
        else:
            if process_started == 1:
                occurrences=1
            else:
                pass
    return True, len_of_chain

def find_longest_chain(array):
    unique_ID = 0 
    process_started = 0
    chain = []
    len_of_chain = []
    for i in range(len(array)):
        if array[i] == 1:
            chain.append(unique_ID)
        else:
            unique_ID+=1
    chain = np.array(chain)
    for unique_val in np.unique(chain):
        len_of_chain.append(sum(chain==unique_val))
    return max(len_of_chain)



isit = is_it_a_full_column(array)
# print(isit)

longest_chain = find_longest_chain(array)
print(longest_chain)