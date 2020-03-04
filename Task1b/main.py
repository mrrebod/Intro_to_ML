#  -*- coding: utf-8 -*-
"""
Introduction to Machine Learning: Task1b

Vukasin Lalic  & Marco Dober aka Snorlax
"""

import numpy as np    

# Read in complete train set from csv file
train_set = np.genfromtxt('train.csv', delimiter=',', skip_header=1)
# Extract ID from train set
Id_train = train_set[:, 0]
# Extract y label from train set
y_train = train_set[:, 1]
# Extract feature from train set
X_train = train_set[:, 2:]
