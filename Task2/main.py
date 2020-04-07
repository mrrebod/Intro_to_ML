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
from sklearn.multiclass import OneVsRestClassifier


# -----------------------------------------------------------------------------

# Read the datasets as DataFrames
test_features_df  = pd.read_csv('test_features.csv')  
train_features_df = pd.read_csv('train_features.csv')  
train_labels_df   = pd.read_csv('train_labels.csv')  

# Convert to numpy
test_features  = test_features_df.to_numpy()
train_features = train_features_df.to_numpy() 
train_labels   = train_labels_df.to_numpy()

# -----------------------------------------------------------------------------

# Create a 3D array where the third dimension is the patient and the first two
# dimesions are all its values and measurements over time (with pid removed)
patients_data = np.zeros((12, train_features.shape[1]-1, int(train_features.shape[0]/12) ))
for i in range(len(train_labels)):
    patients_data[:,:,i] = train_features[i*12 : 12*(i+1) :1, 1:]

# Use a vector to save the corresponding PIDs at the corresponding index
# e.g. pid: 10002 is at index 5, the same as in patients_data[:,:,5]
train_pid = train_labels[:,0]


# -----------------------------------------------------------------------------
# How to handle missing data?  

# Columns with more than half entries missing (6/12) will be dismissed?
# Columns with less than half entries missing will use median values to fill the
# missing entries?
# Maybe Using the impute class

# initialize vectorized patients_data
patients_data_vector = np.zeros((patients_data.shape[2], patients_data.shape[0]*patients_data.shape[1]))

for i in range(patients_data.shape[2]):
    patients_data_try = patients_data[:,:,i]
    
    # If 6 or more entries of a column are not nan and at least one is missing 
    # -> fill the rest with the median
    number_of_not_nans = np.count_nonzero(~np.isnan(patients_data_try), axis=0) 
    columns_to_fill = np.greater_equal(number_of_not_nans, 1) & np.less(number_of_not_nans, 12)
    
    # Compute the median where needed
    where_to_fill  = patients_data_try[:,np.where(columns_to_fill)]
    where_to_fill  = np.squeeze(where_to_fill)   # from 3d to 2d
    median_to_fill = np.nanmedian(where_to_fill,axis=0)
    
    #Find indicies that you need to replace
    inds = np.where(np.isnan(where_to_fill))
    
    #Place column means in the indices. Align the arrays using take
    if np.isscalar(median_to_fill):
        where_to_fill[inds] = median_to_fill
    else:
        where_to_fill[inds] = np.take(median_to_fill, inds[1])
    
    
    # Rewerite the starting array
    patients_data_try[:,np.squeeze(np.where(columns_to_fill))] = where_to_fill
    
   #  patients_data[:,:,i] = patients_data_try
    
    # vectorize patients_data
    patients_data_vector[i,:] = np.ndarray.flatten(patients_data_try)
    
    
# Replace nan values with 0
patients_data_vector = np.nan_to_num(patients_data_vector, nan=0)
    

# What to do with all nan columns or columns where nan is in more than 6 entries?
# -> delete?

# -----------------------------------------------------------------------------

# Train the model 
# clf = LinearSVC(random_state=0, tol=1e-5, class_weight= 'balanced')
clf = OneVsRestClassifier(SVC(kernel='linear', probability=True, class_weight='balanced'))

clf.fit(patients_data_vector, train_labels[:, 1:11])


# for i in range(patients_data.shape[2]):
#     clf.fit(patients_data[:,:,i], train_labels[i,1:])
    

# -----------------------------------------------------------------------------

# Subtask 1
# Predict whether medical tests are ordered by a clinician in the remainder 
# of the hospital stay in the interval [0,1]
# LABEL_BaseExcess, LABEL_Fibrinogen, LABEL_AST, LABEL_Alkalinephos, 
# LABEL_Bilirubin_total, LABEL_Lactate, LABEL_TroponinI, LABEL_SaO2, 
# LABEL_Bilirubin_direct, LABEL_EtCO2

# -----------------------------------------------------------------------------

# Subtask 2
# Predict whether sepsis will occur in the remaining stay in the interval [0,1]
# LABEL_Sepsis

# -----------------------------------------------------------------------------

# Subtask 3
# Predict future mean values of key vital signs
# LABEL_RRate, LABEL_ABPm, LABEL_SpO2, LABEL_Heartrate


