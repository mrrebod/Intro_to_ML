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
from sklearn.linear_model import Ridge
import time
from sklearn.linear_model import LinearRegression

# -----------------------------------------------------------------------------

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

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
# How often complete 12h-colums are nan 

occurance_of_nan = np.zeros(train_features.shape[1])

for i in range(int(train_features.shape[0]/12)):
    nans_per_col = np.count_nonzero(np.isnan(train_features[i*12 : 12*(i+1) :1, 1:]), axis=0)
    which_cols_are_nan = np.where(nans_per_col == 12)
    
    np.add.at(occurance_of_nan, which_cols_are_nan, 1) # np.ufunc.at()

# Convert to percentages
occurance_of_nan = occurance_of_nan/int(train_features.shape[0]/12)*100

mask1 = occurance_of_nan < 90
mask2 = occurance_of_nan >= 90
mask3 = occurance_of_nan >= 95
mask4 = occurance_of_nan <= 10

colums_to_delete = np.where(mask2) # delete all cols with more than 90% missing
colums_to_keep   = np.where(mask4) # keep only cols with less than 10% missing


# Delete the columns: EtCO2, Fibrinogen and Bilirubin_direct
# train_features = np.delete(train_features, colums_to_delete, 1)
# test_features  = np.delete(test_features,colums_to_delete, 1)

# Only keep the frequent ones
train_features = np.squeeze(train_features[:,colums_to_keep])
test_features  = np.squeeze(test_features[:,colums_to_keep]) 

# -----------------------------------------------------------------------------
# Calculate the global mean of each column (except pid)
# global_mean = np.nanmean(train_features[:,1:], axis=0)
global_mean = np.nanmedian(train_features[:,1:], axis=0)

print("Data Setup starts")
tic = time.time()
      
# -----------------------------------------------------------------------------

# Create a 3D array where the third dimension is the patient and the first two
# dimesions are all its values and measurements over time (with pid removed)
patients_data = np.zeros((12, train_features.shape[1]-1, int(train_features.shape[0]/12) ))
patients_orig = np.zeros((12, train_features.shape[1]-1, int(train_features.shape[0]/12) ))
test_data     = np.zeros((12, test_features.shape[1]-1, int(test_features.shape[0]/12) ))
for i in range(len(train_labels)):
    patients_data[:,:,i] = train_features[i*12 : 12*(i+1) :1, 1:]
    patients_orig[:,:,i] = train_features[i*12 : 12*(i+1) :1, 1:]
for i in range(int(len(test_features)/12)):
    test_data[:,:,i]     = test_features[i*12 : 12*(i+1) :1, 1:]

# Use a vector to save the corresponding PIDs at the corresponding index
# e.g. pid: 10002 is at index 5, the same as in patients_data[:,:,5]
train_pid = train_labels[:,0]
test_pid  = test_features[0::12,0] 


# -----------------------------------------------------------------------------
# How to handle missing data?  

# Columns with more than half entries missing (6/12) will be dismissed?
# Columns with less than half entries missing will use median values to fill the
# missing entries?
# Maybe Using the impute class

# initialize vectorized patients_data
patients_data_vector = np.zeros((patients_data.shape[2], patients_data.shape[0]*patients_data.shape[1]))
test_data_vector     = np.zeros((test_data.shape[2], test_data.shape[0]*test_data.shape[1]))
regr = LinearRegression()

for i in range(patients_data.shape[2]):
    patients_data_try = patients_data[:,:,i]
    
    x_axis = patients_data_try[:,0] # Time as x-axis
    
    # If 6 or more entries of a column are not nan and at least one is missing 
    # -> fill the rest with the median
    number_of_not_nans = np.count_nonzero(~np.isnan(patients_data_try), axis=0) 
    columns_to_fill = np.equal(number_of_not_nans, 1)
    columns_to_predict = np.greater_equal(number_of_not_nans, 2) & np.less(number_of_not_nans, 12)
    
    # Which columns consist of only nan entries
    columns_with_only_nan = np.equal(number_of_not_nans, 0)
    
    # Compute the median where needed
    where_to_fill  = patients_data_try[:,np.where(columns_to_fill)]
    where_to_fill  = np.squeeze(where_to_fill)   # from 3d to 2d
    median_to_fill = np.nanmedian(where_to_fill,axis=0)
    
    # Replace the columns with only nan entries by the global means
    patients_data_try[:,np.where(columns_with_only_nan)] = global_mean[np.where(columns_with_only_nan)]
    
    # Not median anymore but the regression prediction
    if np.any(columns_to_predict):
        for index in np.nditer(np.where(columns_to_predict)):
            # Split into nan and not nan parts
            # non nan -> train
            # nan     -> predict
            nan_index     = np.argwhere(np.isnan(patients_data_try[:, index]))
            non_nan_index = np.argwhere(~np.isnan(patients_data_try[:, index]))
            
            regr.fit(x_axis[non_nan_index].reshape(-1,1), patients_data_try[non_nan_index, index])
            patients_data_try[nan_index, index] = regr.predict(x_axis[nan_index].reshape(-1, 1))
            
    # Find indicies that you need to replace
    inds = np.where(np.isnan(where_to_fill))
    
    # Place column means in the indices. Align the arrays using take
    if np.isscalar(median_to_fill):
        where_to_fill[inds] = median_to_fill
    else:
        where_to_fill[inds] = np.take(median_to_fill, inds[1])
        
    
    # # Rewerite the starting array
    patients_data_try[:,np.squeeze(np.where(columns_to_fill))] = where_to_fill

    # vectorize patients_data
    patients_data_vector[i,:] = np.ndarray.flatten(patients_data_try)
    
    
# Same for test data ----------------------------------------------------------
for i in range(test_data.shape[2]):
    test_data_try = test_data[:,:,i]
    
    x_axis = test_data_try[:,0] # Time as x-axis
    
    # If 6 or more entries of a column are not nan and at least one is missing 
    # -> fill the rest with the median
    number_of_not_nans = np.count_nonzero(~np.isnan(test_data_try), axis=0) 
    columns_to_fill = np.equal(number_of_not_nans, 1)
    columns_to_predict = np.greater_equal(number_of_not_nans, 2) & np.less(number_of_not_nans, 12)
    
    # Which columns consist of only nan entries
    columns_with_only_nan = np.equal(number_of_not_nans, 0)
    
    # Compute the median where needed
    where_to_fill  = test_data_try[:,np.where(columns_to_fill)]
    where_to_fill  = np.squeeze(where_to_fill)   # from 3d to 2d
    median_to_fill = np.nanmedian(where_to_fill,axis=0)
    
    # Replace the columns with only nan entries by the global means
    test_data_try[:,np.where(columns_with_only_nan)] = global_mean[np.where(columns_with_only_nan)]
  
    # Not median anymore but the regression prediction
    if np.any(columns_to_predict):
        for index in np.nditer(np.where(columns_to_predict)):
            # Split into nan and not nan parts
            # non nan -> train
            # nan     -> predict
            nan_index     = np.argwhere(np.isnan(test_data_try[:, index]))
            non_nan_index = np.argwhere(~np.isnan(test_data_try[:, index]))
            
            # regr.fit(x_axis[non_nan_index].reshape(-1,1), test_data_try[non_nan_index, index])
            regr.fit(x_axis[non_nan_index].reshape(-1,1), patients_data_try[non_nan_index, index, i]) # fit on patients data
            test_data_try[nan_index, index] = regr.predict(x_axis[nan_index].reshape(-1, 1))
           
            
    # Find indicies that you need to replace
    inds = np.where(np.isnan(where_to_fill))
    
    # Place column means in the indices. Align the arrays using take
    if np.isscalar(median_to_fill):
        where_to_fill[inds] = median_to_fill
    else:
        where_to_fill[inds] = np.take(median_to_fill, inds[1])
        
    # Rewerite the starting array
    test_data_try[:,np.squeeze(np.where(columns_to_fill))] = where_to_fill
    
    # vectorize patients_data
    test_data_vector[i,:] = np.ndarray.flatten(test_data_try)
    
    
# Replace nan values with 0
# patients_data_vector = np.nan_to_num(patients_data_vector, nan=0)
# test_data_vector = np.nan_to_num(test_data_vector, nan=0)

# TODO: Replace nan with mean of TRAIN set
# TODO: Make 1 vector out of the 12 measurements (loose time info...)
toc = time.time()
print("Data Setup done | Duration = ", toc-tic, "seconds")

#%% -----------------------------------------------------------------------------

# Subtask 1
# Predict whether medical tests are ordered by a clinician in the remainder 
# of the hospital stay in the interval [0,1]
# LABEL_BaseExcess, LABEL_Fibrinogen, LABEL_AST, LABEL_Alkalinephos, 
# LABEL_Bilirubin_total, LABEL_Lactate, LABEL_TroponinI, LABEL_SaO2, 
# LABEL_Bilirubin_direct, LABEL_EtCO2

# TODO: Look at hyperparameters of classifier
print("clf  start train and predict")
tic = time.time()

clf = OneVsRestClassifier(SVC(class_weight='balanced'), n_jobs=-1)

# clf.fit(patients_data_vector[0:500,:], train_labels[0:500, 1:11])
clf.fit(patients_data_vector[:,:], train_labels[:, 1:11])

toc = time.time()
print("clf  training done | Duration = ", toc-tic, "seconds")
tic = time.time()

predict_labels = clf.predict(test_data_vector)
predict_distance = clf.decision_function(test_data_vector)
predict_confidence = sigmoid(predict_distance)

toc = time.time()
print("clf  predicting done | Duration = ", toc-tic, "seconds")
# -----------------------------------------------------------------------------

# Subtask 2
# Predict whether sepsis will occur in the remaining stay in the interval [0,1]
# LABEL_Sepsis

# TODO: Look at hyperparameters of classifier
print("clf2 start train and predict")
tic = time.time()

clf_2 = SVC(class_weight='balanced')
clf_2.fit(patients_data_vector[:,:], train_labels[:, 11])

toc = time.time()
print("clf2 training done | Duration = ", toc-tic, "seconds")
tic = time.time()

predict_labels_sepsis = clf_2.predict(test_data_vector)
predict_distance_sepsis = clf_2.decision_function(test_data_vector)
predict_confidence_sepsis = sigmoid(predict_distance_sepsis)

toc = time.time()
print("clf2 predicting done | Duration = ", toc-tic, "seconds")

# -----------------------------------------------------------------------------

# Subtask 3
# Predict future mean values of key vital signs
# LABEL_RRate, LABEL_ABPm, LABEL_SpO2, LABEL_Heartrate

# TODO: Make CrossVal of hyperparameters (Task 1b...)
print("reg  start train and predict")
tic = time.time()

reg = Ridge(alpha=1.0)
reg.fit(patients_data_vector[:,:], train_labels[:, 12:])

toc = time.time()
print("reg  training done | Duration = ", toc-tic, "seconds")
tic = time.time()

predict_reg = reg.predict(test_data_vector)

toc = time.time()
print("reg  predicting done | Duration = ", toc-tic, "seconds")


# -----------------------------------------------------------------------------
# Write the Solution to a csv file

test_pid_array = np.reshape(test_pid, (test_pid.shape[0],1))
predict_confidence_sepsis_array = np.reshape(predict_confidence_sepsis, (predict_confidence_sepsis.shape[0],1))

solution = np.concatenate((predict_confidence, predict_confidence_sepsis_array), axis=1)
solution = np.concatenate((solution, predict_reg), axis=1)
solution = np.concatenate((test_pid_array, solution), axis=1)

solution_df = pd.DataFrame(data   = solution,
                           index  = np.arange(0,solution.shape[0]),
                           columns = train_labels_df.columns)


solution_df.to_csv('prediction.zip', index=False, float_format='%.3f', compression='zip')



