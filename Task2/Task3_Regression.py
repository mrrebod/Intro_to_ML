
"""
Introduction to Machine Learning: Task2

Marco Dober & Vukasin Lalic aka Snorlax

Task 3: Regression 
"""
# %% Import libraries
# General
import pandas as pd
import numpy as np
import time
from tempfile import mkdtemp
from shutil import rmtree
# Model Selection
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression 
# Preprocessing
from sklearn.preprocessing import StandardScaler
# Regressors 
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.kernel_ridge import KernelRidge
# Classifiers
from sklearn.svm import SVC 

# %% Load data from .csv in pd frame and convert to np array

# Read the datasets as DataFrames
test_features_df  = pd.read_csv('test_features.csv')  
train_features_df = pd.read_csv('train_features.csv')  
train_labels_df   = pd.read_csv('train_labels.csv')  

# Convert to numpy
test_features  = test_features_df.to_numpy()
train_features = train_features_df.to_numpy() 
train_labels   = train_labels_df.to_numpy()

# %% Create Feature vector out of the 12*35 measurements
print("Feature vector creation starts")
tic = time.time()

# Calculate the global mean of each column (except pid)
global_mean = np.nanmean(train_features[:,1:], axis=0)

# --------------------------------------------------------    
# Create a 3D array where the third dimension is the patient and the first two
# dimesions are all its values and measurements over time (with pid removed)
# --------------------------------------------------------
# Allocate empty 3D arrays
train_features_3D = np.zeros((12, train_features.shape[1]-1, int(train_features.shape[0]/12)))
test_features_3D  = np.zeros((12, test_features.shape[1]-1, int(test_features.shape[0]/12) ))

# Create 3D array for train and test set
for i in range(len(train_labels)):
    train_features_3D[:,:,i] = train_features[i*12 : 12*(i+1) :1, 1:]
for i in range(int(len(test_features)/12)):
    test_features_3D[:,:,i]  = test_features[i*12 : 12*(i+1) :1, 1:]

# Use a vector to save the corresponding PIDs at the corresponding index
# e.g. pid: 10002 is at index 5, the same as in patients_data[:,:,5]
train_pid = train_labels[:,0]
test_pid  = test_features[0::12,0] 

# Allocate empty feature vector 
# train set
train_features_vector    = np.zeros((train_features_3D.shape[2], 
                                     train_features_3D.shape[0]*train_features_3D.shape[1]))
train_features_vector_v2 = np.zeros((train_features_3D.shape[2], 124))

# test set 
test_features_vector     = np.zeros((test_features_3D.shape[2],
                                     test_features_3D.shape[0]*test_features_3D.shape[1]))
test_features_vector_v2 = np.zeros((test_features_3D.shape[2], 124))

# regressor for imputation
regr = LinearRegression()

# Create feature vector for train set
for i in range(train_features_3D.shape[2]):
    
    # Read out measuremnts from only one patient
    patient_data = train_features_3D[:,:,i]
    
    # Time as x-axis
    x_axis = patient_data[:,0] 
    
    # -> fill the rest with the median
    number_of_not_nans = np.count_nonzero(~np.isnan(patient_data), axis=0) 
    columns_to_fill = np.equal(number_of_not_nans, 1)
    columns_to_predict = np.greater_equal(number_of_not_nans, 2) & np.less(number_of_not_nans, 12)
    
    # Which columns consist of only nan entries
    columns_with_only_nan = np.equal(number_of_not_nans, 0)
    # Which columns consist of only one measurement
    columns_with_one_meas = np.equal(number_of_not_nans, 1)
    
    # Compute the median where needed
    where_to_fill  = patient_data[:,np.where(columns_to_fill)]
    where_to_fill  = np.squeeze(where_to_fill)   # from 3d to 2d
    median_to_fill = np.nanmedian(where_to_fill,axis=0)
    
    # Replace the columns with only nan entries by the global means / 0
    patient_data[:,np.where(columns_with_only_nan)] = 0 #global_mean[np.where(columns_with_only_nan)]
    
    # Not median anymore but the regression prediction
    if np.any(columns_to_predict):
        for index in np.nditer(np.where(columns_to_predict)):
            # Split into nan and not nan parts
            # non nan -> train
            # nan     -> predict
            nan_index     = np.argwhere(np.isnan(patient_data[:, index]))
            non_nan_index = np.argwhere(~np.isnan(patient_data[:, index]))
            
            regr.fit(x_axis[non_nan_index].reshape(-1,1), patient_data[non_nan_index, index])
            patient_data[nan_index, index] = regr.predict(x_axis[nan_index].reshape(-1, 1))
    
    # Find indicies that you need to replace
    inds = np.where(np.isnan(where_to_fill))
    
    # Place column means in the indices. Align the arrays using take
    if np.isscalar(median_to_fill):
        where_to_fill[inds] = median_to_fill
    else:
        where_to_fill[inds] = np.take(median_to_fill, inds[1])
        
        
    # Rewrite the starting array
    patient_data[:,np.squeeze(np.where(columns_to_fill))] = where_to_fill
    
    # Create mean of the 12 measurements  
    patient_data_mean = np.nanmean(patient_data, axis=0)
    
    # Create feature vector with all measurements
    train_features_vector[i,:] = np.ndarray.flatten(patient_data)
    
    # create second version of feature vector with only all measur. from vital signs
    train_features_vector_v2[i, 0:12]    = patient_data[:,0] # time
    train_features_vector_v2[i, 12:17]   = patient_data_mean[1:6] # Age, EtCO2, PTT, BUN, Lactate
    train_features_vector_v2[i, 17:29]   = patient_data[:,6] # Temp
    train_features_vector_v2[i, 29:32]   = patient_data_mean[7:10] # Hgb, HCO3, BaseExcess
    train_features_vector_v2[i, 32:44]   = patient_data[:,10] # RRate
    train_features_vector_v2[i, 44:54]   = patient_data_mean[11:21] # Fibrinogen - Glucose
    train_features_vector_v2[i, 54:66]   = patient_data[:,21] # ABPm
    train_features_vector_v2[i, 66:68]   = patient_data_mean[22:24] # Magnesium, Potassium
    train_features_vector_v2[i, 68:80]   = patient_data[:,24] # ABPd
    train_features_vector_v2[i, 80:82]   = patient_data_mean[25:27] # Calcium, Alkalinephos
    train_features_vector_v2[i, 82:94]   = patient_data[:,27] # SpO2
    train_features_vector_v2[i, 94:97]   = patient_data_mean[28:31] # Bilirubin_direct, Chloride, Hct
    train_features_vector_v2[i, 97:109]  = patient_data[:,31] # Heartrate
    train_features_vector_v2[i, 109:111] = patient_data_mean[32:34] # Bilirubin_total, Troponin1
    train_features_vector_v2[i, 111:123] = patient_data[:,34] # ABPs
    train_features_vector_v2[i, 123]     = patient_data_mean[35] # pH
    
# Create feature vector for test set
for i in range(test_features_3D.shape[2]):
    
    # Read out measuremnts from only one patient
    patient_data = test_features_3D[:,:,i]
    
    # Time as x-axis
    x_axis = patient_data[:,0] 
    
    # -> fill the rest with the median
    number_of_not_nans = np.count_nonzero(~np.isnan(patient_data), axis=0) 
    columns_to_fill = np.equal(number_of_not_nans, 1)
    columns_to_predict = np.greater_equal(number_of_not_nans, 2) & np.less(number_of_not_nans, 12)
    
    # Which columns consist of only nan entries
    columns_with_only_nan = np.equal(number_of_not_nans, 0)
    # Which columns consist of only one measurement
    columns_with_one_meas = np.equal(number_of_not_nans, 1)
    
    # Compute the median where needed
    where_to_fill  = patient_data[:,np.where(columns_to_fill)]
    where_to_fill  = np.squeeze(where_to_fill)   # from 3d to 2d
    median_to_fill = np.nanmedian(where_to_fill,axis=0)
    
    # Replace the columns with only nan entries by the global means
    patient_data[:,np.where(columns_with_only_nan)] = 0 # global_mean[np.where(columns_with_only_nan)]
    
    # Not median anymore but the regression prediction
    if np.any(columns_to_predict):
        for index in np.nditer(np.where(columns_to_predict)):
            # Split into nan and not nan parts
            # non nan -> train
            # nan     -> predict
            nan_index     = np.argwhere(np.isnan(patient_data[:, index]))
            non_nan_index = np.argwhere(~np.isnan(patient_data[:, index]))
            
            regr.fit(x_axis[non_nan_index].reshape(-1,1), patient_data[non_nan_index, index])
            patient_data[nan_index, index] = regr.predict(x_axis[nan_index].reshape(-1, 1))
    
    # Find indicies that you need to replace
    inds = np.where(np.isnan(where_to_fill))
    
    # Place column means in the indices. Align the arrays using take
    if np.isscalar(median_to_fill):
        where_to_fill[inds] = median_to_fill
    else:
        where_to_fill[inds] = np.take(median_to_fill, inds[1])
          
    # Rewerite the starting array
    patient_data[:,np.squeeze(np.where(columns_to_fill))] = where_to_fill
    
    # Create mean of the 12 measurements  
    patient_data_mean = np.nanmean(patient_data, axis=0)
    
    # Create feature vector with all measurements
    test_features_vector[i,:] = np.ndarray.flatten(patient_data)
    
    # create second version of feature vector with only all measur. from vital signs
    test_features_vector_v2[i, 0:12]    = patient_data[:,0] # time
    test_features_vector_v2[i, 12:17]   = patient_data_mean[1:6] # Age, EtCO2, PTT, BUN, Lactate
    test_features_vector_v2[i, 17:29]   = patient_data[:,6] # Temp
    test_features_vector_v2[i, 29:32]   = patient_data_mean[7:10] # Hgb, HCO3, BaseExcess
    test_features_vector_v2[i, 32:44]   = patient_data[:,10] # RRate
    test_features_vector_v2[i, 44:54]   = patient_data_mean[11:21] # Fibrinogen - Glucose
    test_features_vector_v2[i, 54:66]   = patient_data[:,21] # ABPm
    test_features_vector_v2[i, 66:68]   = patient_data_mean[22:24] # Magnesium, Potassium
    test_features_vector_v2[i, 68:80]   = patient_data[:,24] # ABPd
    test_features_vector_v2[i, 80:82]   = patient_data_mean[25:27] # Calcium, Alkalinephos
    test_features_vector_v2[i, 82:94]   = patient_data[:,27] # SpO2
    test_features_vector_v2[i, 94:97]   = patient_data_mean[28:31] # Bilirubin_direct, Chloride, Hct
    test_features_vector_v2[i, 97:109]  = patient_data[:,31] # Heartrate
    test_features_vector_v2[i, 109:111] = patient_data_mean[32:34] # Bilirubin_total, Troponin1
    test_features_vector_v2[i, 111:123] = patient_data[:,34] # ABPs
    test_features_vector_v2[i, 123]     = patient_data_mean[35] # pH    

X_train    = train_features_vector
X_train_v2 = train_features_vector_v2
X_test     = test_features_vector
X_test_v2  = test_features_vector_v2

toc = time.time()
print("Feature vectorization finisched | Duration = ", toc-tic, "seconds")



# %% Task2 

X_train_new = SelectKBest(k=10).fit_transform(X_train, train_labels[:, 11])


pipe_t2 = Pipeline([#('scaler' , stand_scaler),
                    ('clf'    , SVC(class_weight='balanced'))]
                   )

param_grid_t2 = dict(clf__C     = [ 1], 
                     clf__gamma = ['scale'])



grid_search_t2 = GridSearchCV(pipe_t2, param_grid_t2, scoring='balanced_accuracy')

print("grid_search Task2 started")
tic = time.time()
grid_search_t2.fit(X_train_new, train_labels[:, 11])
toc = time.time()
print("grid_search Task2 finisched | Duration = ", toc-tic, "seconds")

print('!!!!!!!!!!!!! Best Params for Task2 !!!!!!!!!!!!!!!!!!!')
print(grid_search_t2.best_params_)
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
print('!!!!!!!!!!!!! Best CV Score for Task2 !!!!!!!!!!!!!!!!!!!')
print(grid_search_t2.best_score_)
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

# %% Task3:

# %% Selcet features 
selected_features = [] 
for i in range(12, 16):
    selector = SelectKBest(f_regression, k='all')
    selector.fit(X_train_v2, train_labels[:,i])
    selected_features.append(list(selector.scores_))


# Select feature with highest scores 
threshold_mean = 1000 # Chosen after investigation of mean values
threshold_max = 2000
selected_features_thresholded = np.mean(selected_features, axis=0) > threshold_mean
# MaxCS
#selected_features_thresholded = np.max(selected_features, axis=0) > threshold_max

X_train_v2_new = X_train_v2[:, np.where(selected_features_thresholded)]
X_train_v2 = X_train_v2_new[:,0,:]

# %% Create pipeline components used in task 3
# Create standard scaler
stand_scaler    = StandardScaler()
# Regression for task3
ridge_regressor = Ridge(fit_intercept=True)
# store transformers and reuse when possible
cachedir = mkdtemp()

pipe_t3 = Pipeline([#('selector', feat_select),
                    ('scaler', stand_scaler),
                    ('regressor', ridge_regressor)],
                     memory=cachedir)

# Create grid with all hperparameters
param_grid_t3 = dict(scaler=['passthrough', stand_scaler],
                     #regressor=[ridge_regressor, KernelRidge(kernel='rbf')],
                     regressor__alpha=[1, 10, 100, 1000, 10000, 150000],
                     regressor__fit_intercept = [True, False]
                     )

# Make grid search for best paramters with 5-fold CrossVal
grid_search_t3 = GridSearchCV(pipe_t3, param_grid_t3, scoring='r2')

print("grid_search X_train_v2 started")
tic = time.time()
grid_search_t3.fit(X_train_v2, train_labels[:, 12:])
toc = time.time()
print("grid_search  finisched | Duration = ", toc-tic, "seconds")

print('!!!!!!!!!!!!! Best Params for Task3 !!!!!!!!!!!!!!!!!!!')
print(grid_search_t3.best_params_)
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
print('!!!!!!!!!!!!! Best CV Score for Task3 !!!!!!!!!!!!!!!!!!!')
print(grid_search_t3.best_score_)
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')






