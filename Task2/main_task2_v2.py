"""
Introduction to Machine Learning: Task2

Marco Dober & Vukasin Lalic aka Snorlax
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


#%%-----------------------------------------------------------------------------

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
# Calculate the global mean of each column (except pid)
global_mean = np.nanmean(train_features[:,1:], axis=0)

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

# initialize vectorized patients_data
patients_data_vector = np.zeros((patients_data.shape[2], patients_data.shape[0]*patients_data.shape[1]))
test_data_vector     = np.zeros((test_data.shape[2], test_data.shape[0]*test_data.shape[1]))

patients_data_vector_v2 = np.zeros((patients_data.shape[2], 124))

# vector which counts all nan columns
number_all_nan_columns = np.zeros(patients_data.shape[1])
# vector which counts all measurments done exacly once
number_one_meas_columns = np.zeros(patients_data.shape[1]) 

regr = LinearRegression()


for i in range(patients_data.shape[2]):
    patients_data_try = patients_data[:,:,i]
    
    # Time as x-axis
    x_axis = patients_data_try[:,0] 
    
    # -> fill the rest with the median
    number_of_not_nans = np.count_nonzero(~np.isnan(patients_data_try), axis=0) 
    columns_to_fill = np.equal(number_of_not_nans, 1)
    columns_to_predict = np.greater_equal(number_of_not_nans, 2) & np.less(number_of_not_nans, 12)
    
    # Which columns consist of only nan entries
    columns_with_only_nan = np.equal(number_of_not_nans, 0)
    # Which columns consist of only one measurement
    columns_with_one_meas = np.equal(number_of_not_nans, 1)
    
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
        
    
    # Rewerite the starting array
    patients_data_try[:,np.squeeze(np.where(columns_to_fill))] = where_to_fill
    
    # Calculate the means of the patients_data_try
    patients_data_try_mean = np.mean(patients_data_try, axis=0)
    
    # vectorize patients_data
    patients_data_vector[i,:] = np.ndarray.flatten(patients_data_try)
    
    # create second version of feature vector
    patients_data_vector_v2[i, 0:12] = patients_data_try[:,0] # time
    patients_data_vector_v2[i, 12:17]   = patients_data_try_mean[1:6] # Age, EtCO2, PTT, BUN, Lactate
    patients_data_vector_v2[i, 17:29]   = patients_data_try[:,6] # Temp
    patients_data_vector_v2[i, 29:32]   = patients_data_try_mean[7:10] # Hgb, HCO3, BaseExcess
    patients_data_vector_v2[i, 32:44]   = patients_data_try[:,10] # RRate
    patients_data_vector_v2[i, 44:54]   = patients_data_try_mean[11:21] # Fibrinogen - Glucose
    patients_data_vector_v2[i, 54:66]   = patients_data_try[:,21] # ABPm
    patients_data_vector_v2[i, 66:68]   = patients_data_try_mean[22:24] # Magnesium, Potassium
    patients_data_vector_v2[i, 68:80]   = patients_data_try[:,24] # ABPd
    patients_data_vector_v2[i, 80:82]   = patients_data_try_mean[25:27] # Calcium, Alkalinephos
    patients_data_vector_v2[i, 82:94]   = patients_data_try[:,27] # SpO2
    patients_data_vector_v2[i, 94:97]   = patients_data_try_mean[28:31] # Bilirubin_direct, Chloride, Hct
    patients_data_vector_v2[i, 97:109]   = patients_data_try[:,31] # Heartrate
    patients_data_vector_v2[i, 109:111]   = patients_data_try_mean[32:34] # Bilirubin_total, Troponin1
    patients_data_vector_v2[i, 111:123]   = patients_data_try[:,34] # ABPs
    patients_data_vector_v2[i, 123]   = patients_data_try_mean[35] # pH
    
    
    # Extract statistics of measurement occurancy 
    # increment number_of_all_nan_columns at the column where only nan values are
    np.add.at(number_all_nan_columns, np.where(columns_with_only_nan), 1)
    # increment number_one_meas_columns at the column where only one meas is
    np.add.at(number_one_meas_columns, np.where(columns_with_one_meas), 1)
   
    
# calculate percentage of all nan columns
percent_all_nan_columns = number_all_nan_columns/patients_data.shape[2]*100
# create panda frame for percent_all_nan_columns
percent_all_nan_columns_df = pd.DataFrame(data    = percent_all_nan_columns.reshape(1,36), 
                                          columns = train_features_df.columns[1:])

# calculate percentage of one meas only
percent_one_meas_columns = number_one_meas_columns/patients_data.shape[2]*100
# create panda frame for number_one_meas_columns
percent_one_meas_columns_df = pd.DataFrame(data    = percent_one_meas_columns.reshape(1,36), 
                                          columns = train_features_df.columns[1:])
 

test_data_vector_v2 = np.zeros((test_data.shape[2], 124))
   
# Same for test data ----------------------------------------------------------
for i in range(test_data.shape[2]):
    test_data_try = test_data[:,:,i]
    
    # If 6 or more entries of a column are not nan and at least one is missing 
    # -> fill the rest with the median
    number_of_not_nans = np.count_nonzero(~np.isnan(test_data_try), axis=0) 
    columns_to_fill = np.greater_equal(number_of_not_nans, 1) & np.less(number_of_not_nans, 12)
    
    # Which columns consist of only nan entries
    columns_with_only_nan = np.equal(number_of_not_nans, 0)
    
    # Compute the median where needed
    where_to_fill  = test_data_try[:,np.where(columns_to_fill)]
    where_to_fill  = np.squeeze(where_to_fill)   # from 3d to 2d
    median_to_fill = np.nanmedian(where_to_fill,axis=0)
    
    # Replace the columns with only nan entries by the global means
    test_data_try[:,np.where(columns_with_only_nan)] = global_mean[np.where(columns_with_only_nan)]
  
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
    
    # Calculate the means of the test_data_try
    test_data_try_mean = np.mean(test_data_try, axis=0)
    
    # create second version of feature vector
    test_data_vector_v2[i, 0:12] = test_data_try[:,0] # time
    test_data_vector_v2[i, 12:17]   = test_data_try_mean[1:6] # Age, EtCO2, PTT, BUN, Lactate
    test_data_vector_v2[i, 17:29]   = test_data_try[:,6] # Temp
    test_data_vector_v2[i, 29:32]   = test_data_try_mean[7:10] # Hgb, HCO3, BaseExcess
    test_data_vector_v2[i, 32:44]   = test_data_try[:,10] # RRate
    test_data_vector_v2[i, 44:54]   = test_data_try_mean[11:21] # Fibrinogen - Glucose
    test_data_vector_v2[i, 54:66]   = test_data_try[:,21] # ABPm
    test_data_vector_v2[i, 66:68]   = test_data_try_mean[22:24] # Magnesium, Potassium
    test_data_vector_v2[i, 68:80]   = test_data_try[:,24] # ABPd
    test_data_vector_v2[i, 80:82]   = test_data_try_mean[25:27] # Calcium, Alkalinephos
    test_data_vector_v2[i, 82:94]   = test_data_try[:,27] # SpO2
    test_data_vector_v2[i, 94:97]   = test_data_try_mean[28:31] # Bilirubin_direct, Chloride, Hct
    test_data_vector_v2[i, 97:109]   = test_data_try[:,31] # Heartrate
    test_data_vector_v2[i, 109:111]   = test_data_try_mean[32:34] # Bilirubin_total, Troponin1
    test_data_vector_v2[i, 111:123]   = test_data_try[:,34] # ABPs
    test_data_vector_v2[i, 123]   = test_data_try_mean[35] # pH
    
    
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

# clf = OneVsRestClassifier(SVC(class_weight='balanced'), n_jobs=2)
clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=500,random_state=143), n_jobs=2)
# clf = OneVsRestClassifier(MLPClassifier(alpha=0.00001, random_state=0), n_jobs=2)

# clf.fit(patients_data_vector[0:500,:], train_labels[0:500, 1:11])
clf.fit(patients_data_vector_v2, train_labels[:, 1:11])

toc = time.time()
print("clf  training done | Duration = ", toc-tic, "seconds")
tic = time.time()

# predict_labels = clf.predict(test_data_vector_v2)
# predict_distance = clf.decision_function(test_data_vector_v2)
# predict_confidence = sigmoid(predict_distance)

predict_confidence = clf.predict_proba(test_data_vector_v2)


toc = time.time()
print("clf  predicting done | Duration = ", toc-tic, "seconds")
# -----------------------------------------------------------------------------

# Subtask 2
# Predict whether sepsis will occur in the remaining stay in the interval [0,1]
# LABEL_Sepsis

# TODO: Look at hyperparameters of classifier
print("clf2 start train and predict")
tic = time.time()

# clf_2 = SVC(class_weight='balanced')
clf_2 = RandomForestClassifier(n_estimators=500,random_state=143)
# clf_2 = MLPClassifier(alpha=0.00001, random_state=0)

clf_2.fit(patients_data_vector_v2, train_labels[:, 11])


toc = time.time()
print("clf2 training done | Duration = ", toc-tic, "seconds")
tic = time.time()

# predict_labels_sepsis = clf_2.predict(test_data_vector_v2)
# predict_distance_sepsis = clf_2.decision_function(test_data_vector_v2)
# predict_confidence_sepsis = sigmoid(predict_distance_sepsis)

predict_confidence_sepsis = clf_2.predict_proba(test_data_vector_v2)[:,1]

toc = time.time()
print("clf2 predicting done | Duration = ", toc-tic, "seconds")

#%% -----------------------------------------------------------------------------

# Standardize data
scaler = StandardScaler()

reg = Ridge(alpha=100000, fit_intercept=True)

# Create pipeline
pipe = Pipeline([
    ('scaler', scaler),
    ('regress', reg)
    ])

# Hyperparameters to evaluate best model 
# param_grid = dict()
param_grid = {  'scaler': ['passthrough'],
                'regress__alpha': [100000, 200000, 500000, 1000000]}

# Make grid search for best model
grid_search = GridSearchCV(pipe, param_grid, scoring='r2', cv=3)

grid_search.fit(patients_data_vector_v2, train_labels[:, 12:])

print('!!!!!!!!!!!!! All Results !!!!!!!!!!!!!!!!!!!')
print('Rank|Result')
print(grid_search.cv_results_['rank_test_score'], grid_search.cv_results_['mean_test_score'])
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
print('!!!!!!!!!!!!! Best Params !!!!!!!!!!!!!!!!!!!')
print(grid_search.best_params_)
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
print('!!!!!!!!!!!!! Best CV Score !!!!!!!!!!!!!!!!!')
print(grid_search.best_score_)
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

cv_result_df = pd.DataFrame(grid_search.cv_results_)

# %%

# Subtask 3
# Predict future mean values of key vital signs
# LABEL_RRate, LABEL_ABPm, LABEL_SpO2, LABEL_Heartrate

# Check if maybe MultioutputRegressor or RegressorChain improves performance 

print("reg  start train and predict")
tic = time.time()

# Standardize data
scaler = StandardScaler()

reg = Ridge(alpha=200000, fit_intercept=True)

# Standardize data 
# patients_data_vector_v2_stand = scaler.fit_transform(patients_data_vector_v2)
# test_data_vector_v2_stand = scaler.transform(test_data_vector_v2)

patients_data_vector_v2_stand = patients_data_vector_v2
test_data_vector_v2_stand = test_data_vector_v2

reg.fit(patients_data_vector_v2_stand, train_labels[:, 12:])

toc = time.time()
print("reg  training done | Duration = ", toc-tic, "seconds")
tic = time.time()

predict_reg = reg.predict(test_data_vector_v2_stand)

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
