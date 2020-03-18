#  -*- coding: utf-8 -*-
"""
Introduction to Machine Learning: Task1b

Vukasin Lalic  & Marco Dober aka Snorlax
"""

import numpy as np  
from sklearn.linear_model import RidgeCV   
from sklearn.preprocessing import StandardScaler

# Read in complete train set from csv file
train_set = np.genfromtxt('train.csv', delimiter=',', skip_header=1)
# Extract ID from train set
Id_train = train_set[:, 0]
# Extract y label from train set
y_train = train_set[:, 1]
# Extract feature from train set
X_train = train_set[:, 2:]

# Standardize Data
# initialize scaler 
scaler = StandardScaler()
# applay standardization
X_train_stand = scaler.fit_transform(X_train)
# X_train_stand = X_train

# compute Phi with standardized features 
Phi_train_stand = np.zeros((np.shape(X_train)[0], 21))
# Phi 0-4
Phi_train_stand[:, 0:5] = X_train_stand
# Phi 5-9
Phi_train_stand[:, 5:10] = np.square(X_train_stand)
# Phi 10-14
Phi_train_stand[:, 10:15] = np.exp(X_train_stand)
# Phi 15-19
Phi_train_stand[:, 15:20] = np.cos(X_train_stand)
# Phi 21
Phi_train_stand[:, 20] = np.ones(np.shape(X_train_stand)[0])

# Compute ridge regression for standardized data 
clf = RidgeCV(alphas=[0.1, 1, 10, 100, 1000, 10000], fit_intercept=False)
clf.fit(Phi_train_stand, y_train)

# save coefficients into array
weights_ridge_stand = clf.coef_

# print some parameter of model
print(weights_ridge_stand)
print(clf.alpha_)
print(clf.score(Phi_train_stand, y_train))

# save data
np.savetxt('submission.csv', weights_ridge_stand)