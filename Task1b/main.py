#  -*- coding: utf-8 -*-
"""
Introduction to Machine Learning: Task1b

Vukasin Lalic  & Marco Dober aka Snorlax
"""

import numpy as np  
from sklearn.linear_model import LinearRegression
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

"""
# Compute linear regression 
reg = LinearRegression(fit_intercept=False)
reg.fit(Phi_train, y_train)

y_predict = reg.predict(Phi_train)
weights = reg.coef_
print(weights)
print(reg.score(Phi_train,y_train))
# write to submission file 
#np.savetxt('submission.csv', weights)
"""

"""
#compute weights with closed form 
h = np.matmul(np.linalg.inv(np.matmul(np.transpose(Phi_train), Phi_train)),np.transpose(Phi_train))
weights_closed = np.matmul(h,y_train)

condition = np.linalg.cond(np.matmul(np.transpose(Phi_train), Phi_train))
""" 
"""
# Make Ride regression 
clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 10, 100],fit_intercept=False).fit(Phi_train, y_train)
print(clf.score(Phi_train, y_train))
print(clf.alpha_)
weights_ridge = clf.coef_
y_preditc_ridge = clf.predict(Phi_train)
np.savetxt('submission.csv', weights_ridge)
"""

# Standardize Data
# initialize scaler 
scaler = StandardScaler()
# applay standardization
X_train_norm = scaler.fit_transform(X_train)

# compute Phi with standardized features 
Phi_train_norm = np.zeros((np.shape(X_train)[0],21))
# Phi 0-4
Phi_train_norm[:,0:5] = X_train
# Phi 5-9
Phi_train_norm[:,5:10] = np.square(X_train_norm)
# Phi 10-14
Phi_train_norm[:,10:15] = np.exp(X_train_norm)
# Phi 15-19
Phi_train_norm[:,15:20] = np.cos(X_train_norm)
# Phi 21
Phi_train_norm[:,20] = np.ones(np.shape(X_train_norm)[0])

# Compute ridge regression for standardized data 
clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 10, 100],fit_intercept=False).fit(Phi_train_norm, y_train)
clf.fit(Phi_train_norm, y_train)

weights_ridge_norm = clf.coef_
print(weights_ridge_norm)
print(clf.score(Phi_train_norm,y_train))
# save data
np.savetxt('submission.csv', weights_ridge_norm)