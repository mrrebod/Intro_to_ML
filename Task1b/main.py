#  -*- coding: utf-8 -*-
"""
Introduction to Machine Learning: Task1b

Vukasin Lalic  & Marco Dober aka Snorlax
"""

import numpy as np  
from sklearn.linear_model import LinearRegression   

# Read in complete train set from csv file
train_set = np.genfromtxt('train.csv', delimiter=',', skip_header=1)
# Extract ID from train set
Id_train = train_set[:, 0]
# Extract y label from train set
y_train = train_set[:, 1]
# Extract feature from train set
X_train = train_set[:, 2:]

# Generat Phi vector 
Phi_train = np.zeros((np.shape(X_train)[0],21))
# Phi 0-4
Phi_train[:,0:5] = X_train
# Phi 5-9
Phi_train[:,5:10] = np.square(X_train)
# Phi 10-14
Phi_train[:,10:15] = np.exp(X_train)
# Phi 15-19
Phi_train[:,15:20] = np.cos(X_train)
# Phi 21
Phi_train[:,20] = np.ones(np.shape(X_train)[0])

# Compute linear regression 
reg = LinearRegression(fit_intercept=False)
reg.fit(Phi_train, y_train)

weights = reg.coef_
print(weights)
print(reg.score(Phi_train,y_train))

# write to submission file 
np.savetxt('submission.csv', weights)
