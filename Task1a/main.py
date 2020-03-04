#  -*- coding: utf-8 -*-
"""
Introduction to Machine Learning: Task1a

Vukasin Lalic  & Marco Dober aka Teamsnorlax
"""

import numpy as np    
from sklearn.model_selection import KFold 
from sklearn.linear_model import LinearRegression

# Read in complete train set from csv file
data_set = np.genfromtxt('train.csv', delimiter=',', skip_header=1)
# Extract ID from train set
Id = data_set[:, 0]
# Extract y label from train set
y = data_set[:, 1]
# Extract feature from train set
X = data_set[:, 2:]

# Split data set into train and test set
kf = KFold(n_splits=10)
print('Number of splits: ', kf.get_n_splits(X))
train = kf.split(X)
# print('Split indices', kf.split(X))
# fancy comment





