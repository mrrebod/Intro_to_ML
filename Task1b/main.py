#  -*- coding: utf-8 -*-
"""
Introduction to Machine Learning: Task1b

Vukasin Lalic  & Marco Dober aka Snorlax
"""

import numpy as np  
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV   

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

y_predict = reg.predict(Phi_train)
weights = reg.coef_
print(weights)
print(reg.score(Phi_train,y_train))

#compute weights with closed form 
h = np.matmul(np.linalg.inv(np.matmul(np.transpose(Phi_train), Phi_train)),np.transpose(Phi_train))
weights_closed = np.matmul(h,y_train)

condition = np.linalg.cond(np.matmul(np.transpose(Phi_train), Phi_train))

# write to submission file 
#np.savetxt('submission.csv', weights)

# Make Ride regression 
clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 10, 100],fit_intercept=False).fit(Phi_train, y_train)
print(clf.score(Phi_train, y_train))
print(clf.alpha_)
weights_ridge = clf.coef_
y_preditc_ridge = clf.predict(Phi_train)
np.savetxt('submission.csv', weights_ridge)

"""
# Normalize Data: Standardization 
X_train_mean = np.mean(X_train, axis=0) 
X_train_var = np.var(X_train, axis=0)

X_train_norm = np.divide((X_train - X_train_mean), X_train_var)

# compute Phi with normalized features 
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

# Compute linear regression for normailize data 
reg_norm = LinearRegression(fit_intercept=False)
reg_norm.fit(Phi_train_norm, y_train)

weights_norm = reg_norm.coef_
print(weights_norm)
print(reg_norm.score(Phi_train_norm,y_train))
"""
 