"""
Introduction to Machine Learning: Task2

Marco Dober & Vukasin Lalic aka Snorlax
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score  # The used score (not needed?)
from sklearn.svm import LinearSVC          # Let's start with the linear one
from sklearn.svm import SVC                # Try out later
from sklearn.impute import SimpleImputer   # Maybe use this for incomplete data


# Read the datasets as DataFrames
test_features_df  = pd.read_csv('test_features.csv')  
train_features_df = pd.read_csv('train_features.csv')  
train_labels_df   = pd.read_csv('train_labels.csv')  

# Convert to numpy
test_features  = test_features_df.to_numpy()
train_features = train_features_df.to_numpy() 
train_labels   = train_labels_df.to_numpy()

# Create a 3D array where the third dimension is the patient and the first two
# dimesions are all its values and measurements over time
patients_data = np.zeros((12, train_features.shape[1], int(train_features.shape[0]/12) ))
for i in range(len(train_labels)):
    patients_data[:,:,i] = train_features[i*12 : 12*(i+1) :1, :]
# Maybe remove the pid from the entires?

# Use a vector to save the corresponding PIDs at the corresponding index
# e.g. pid: 10002 is at index 5, the same as in patients_data[:,:,5]
train_pid = train_labels[:,0]

# How to handle missing data? 
# Columns with more than half entries missing (6/12) will be dismissed?
# Columns with less than half entries missing will use median values to fill the
# missing entries?
# Maybe Using the impute class



# Subtask 1
# Predict whether medical tests are ordered by a clinician in the remainder 
# of the hospital stay in the interval [0,1]
# LABEL_BaseExcess, LABEL_Fibrinogen, LABEL_AST, LABEL_Alkalinephos, 
# LABEL_Bilirubin_total, LABEL_Lactate, LABEL_TroponinI, LABEL_SaO2, 
# LABEL_Bilirubin_direct, LABEL_EtCO2


# Subtask 2
# Predict whether sepsis will occur in the remaining stay in the interval [0,1]
# LABEL_Sepsis


# Subtask 3
# Predict future mean values of key vital signs
# LABEL_RRate, LABEL_ABPm, LABEL_SpO2, LABEL_Heartrate


