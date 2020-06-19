# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 16:57:32 2020

@author: Vukas
"""
# General
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
# Classifier
from sklearn.multiclass import OneVsRestClassifier

from sklearn.ensemble import RandomForestClassifier

# Regressors 
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
# Selection
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold





print("reg  start train and predict")
tic = time.time()

# Standardize data
scaler = StandardScaler()

reg = Ridge(alpha=100000, fit_intercept=True)

# Create pipeline
pipe = Pipeline([
    ('scaler', scaler),
    ('regress', reg)
    # ('mlp', MLPClassifier(max_iter=200))
    ])

# Hyperparameters to evaluate best model 
# param_grid = dict()
param_grid = {  'scaler': ['passthrough', scaler],
                'regress__alpha': [1000, 10000, 50000, 100000],
                'regress__fit_intercept': [True,False]}

# Make grid search for best model
grid_search = GridSearchCV(pipe, param_grid, scoring='r2', cv=3)


# Standardize data 
patients_data_vector_v2_stand = scaler.fit_transform(patients_data_vector_v2)
test_data_vector_v2_stand = scaler.transform(test_data_vector_v2)

reg.fit(patients_data_vector_v2_stand, train_labels[:, 12:])

toc = time.time()
print("reg  training done | Duration = ", toc-tic, "seconds")
tic = time.time()

predict_reg = reg.predict(test_data_vector_v2_stand)

toc = time.time()
print("reg  predicting done | Duration = ", toc-tic, "seconds")


