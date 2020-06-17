# -*- coding: utf-8 -*-
"""
Introduction to Machine Learning: Task4

Marco Dober & Vukasin Lalic aka Snorlax
"""

# %% Import libraries 
# General stuff 
import numpy as np 
import pandas as pd
import time


# %% Read in data sets and convert to numpy arrays 

# Read the datasets as DataFrames
test_triplets_df  = pd.read_csv('test_triplets.txt', header=None, names=['a','b','c'], delim_whitespace=True)  
train_triplets_df = pd.read_csv('train_triplets.txt', header=None, names=['a','b','c'], delim_whitespace=True)  

# Convert to numpy
test_triplets  = test_triplets_df.to_numpy()
train_triplets = train_triplets_df.to_numpy() 

# %% 




# %% Save test labels to csv
# np.savetxt('submission.csv', test_labels, fmt='%1.0f')