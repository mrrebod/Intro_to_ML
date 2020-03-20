#  -*- coding: utf-8 -*-
"""
Introduction to Machine Learning: Task1b

Vukasin Lalic  & Marco Dober aka Snorlax
"""

import numpy as np  
from sklearn.linear_model import RidgeCV   

# Read in complete train set from csv file
train_set = np.genfromtxt('train.csv', delimiter=',', skip_header=1)
# Extract ID from train set
Id_train = train_set[:, 0]
# Extract y label from train set
y_train = train_set[:, 1]
# Extract feature from train set
X_train = train_set[:, 2:]

# compute Phi  
Phi_train = np.zeros((np.shape(X_train)[0], 21))
# Phi 0-4
Phi_train[:, 0:5] = X_train
# Phi 5-9
Phi_train[:, 5:10] = np.square(X_train)
# Phi 10-14
Phi_train[:, 10:15] = np.exp(X_train)
# Phi 15-19
Phi_train[:, 15:20] = np.cos(X_train)
# Phi 21
Phi_train[:, 20] = np.ones(np.shape(X_train)[0])

# Compute ridge regression
clf_Ridge = RidgeCV(alphas=[0.1, 1, 10, 100, 1000, 10000], fit_intercept=False, cv=10)
clf_Ridge.fit(Phi_train, y_train)

# save coefficients into array
weights_Ridge = clf_Ridge.coef_

# print some parameter of model
print('Ridge weights: ', weights_Ridge)
print('Ridge alpha: ', clf_Ridge.alpha_)
print('Ridge score: ', clf_Ridge.score(Phi_train, y_train))

# save data
np.savetxt('submission.csv', weights_Ridge)