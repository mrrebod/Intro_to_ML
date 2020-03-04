#  -*- coding: utf-8 -*-
"""
Introduction to Machine Learning: Task1a

Vukasin Lalic  & Marco Dober aka Snorlax
"""

import numpy as np    
from sklearn.model_selection import KFold 
from sklearn.linear_model import Ridge

# Read in complete train set from csv file
data_set = np.genfromtxt('train.csv', delimiter=',', skip_header=1)
# Extract ID from train set
Id = data_set[:, 0]
# Extract y label from train set
y = data_set[:, 1]
# Extract feature from train set
X = data_set[:, 2:]

# lambda
lamb = np.array([0.01, 0.1, 1, 10, 100])

# Split data set into train and test set
kf = KFold(n_splits=10)
print('Number of splits: ', kf.get_n_splits(X))


rmse_avg = np.zeros(np.shape(lamb))

for i in range(len(lamb)):
    clf = Ridge(lamb[i])
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        clf.fit(X_train,y_train)
        y_predict = clf.predict(X_test)
        
        rmse_batch = np.sqrt(np.mean((y_test-y_predict)**2))
        
        rmse_avg[i] = rmse_avg[i] + (rmse_batch)/10
        
        
np.savetxt('submission.csv',rmse_avg)

    





