"""
Introduction to Machine Learning: Task2

Marco Dober & Vukasin Lalic aka Snorlax

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
# Preprocessing
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
# Regressors 
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
# Classifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC 

# %% Definition of own functions

# Sigmoid
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

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

X_train    = train_features_vector[0:2000,:]
X_train_v2 = train_features_vector_v2[0:2000,:]
X_test     = test_features_vector
X_test_v2  = test_features_vector_v2

toc = time.time()
print("Feature vectorization finisched | Duration = ", toc-tic, "seconds")

# %% Create pipeline components used in task 1, 2 & 3
# Create KNN imputer 
knn_imputer     = KNNImputer()
# Create simple imputer 
simple_imputer  = SimpleImputer()
# Create standard scaler
stand_scaler    = StandardScaler()
# Classifier for task1
clf_t1 = OneVsRestClassifier(SVC(class_weight='balanced'), n_jobs=2)
# Classifier for task2
clf_t2 = SVC(class_weight='balanced')
# Regression for task3
ridge_regressor = Ridge(fit_intercept=False)
# store transformers and reuse when possible
cachedir = mkdtemp()

# %% TASK 1: 
 
"""
# Put everything into pipeline
pipe_t1 = Pipeline([('imputer', knn_imputer),
                    ('scaler' , stand_scaler),
                    ('clf'    , clf_t1)],
                    memory=cachedir)
"""

# Put everything into pipeline
pipe_t1 = Pipeline([#('scaler' , stand_scaler),
                    ('clf'    , clf_t1)],
                    memory=cachedir)

"""    
# Create grid with all hperparameters
param_grid_t1 = dict(imputer = [KNNImputer(), SimpleImputer()],
                     clf     = [OneVsRestClassifier(SVC(class_weight='balanced'), n_jobs=2),
                                OneVsRestClassifier(SVC(C=0.1, class_weight='balanced'), n_jobs=2),
                                OneVsRestClassifier(SVC(C=100, class_weight='balanced'), n_jobs=2),
                                OneVsRestClassifier(SVC(C=0.1, gamma=0.1, class_weight='balanced'), n_jobs=2),
                                OneVsRestClassifier(SVC(C=100, gamma=0.1, class_weight='balanced'), n_jobs=2),
                                OneVsRestClassifier(SVC(C=0.1, gamma=10, class_weight='balanced'), n_jobs=2),
                                OneVsRestClassifier(SVC(C=100, gamma=10, class_weight='balanced'), n_jobs=2)])

"""

# Create grid with all hperparameters
param_grid_t1 = dict(clf = [OneVsRestClassifier(SVC(class_weight='balanced'), n_jobs=2),
                            OneVsRestClassifier(SVC(C=0.1, class_weight='balanced'), n_jobs=2),
                            OneVsRestClassifier(SVC(C=100, class_weight='balanced'), n_jobs=2),
                            OneVsRestClassifier(SVC(C=0.1, gamma=0.1, class_weight='balanced'), n_jobs=2),
                            OneVsRestClassifier(SVC(C=100, gamma=0.1, class_weight='balanced'), n_jobs=2),
                            OneVsRestClassifier(SVC(C=0.1, gamma=10, class_weight='balanced'), n_jobs=2),
                            OneVsRestClassifier(SVC(C=100, gamma=10, class_weight='balanced'), n_jobs=2)])

grid_search_t1 = GridSearchCV(pipe_t1, param_grid_t1)

print("grid_search Task1 started")
tic = time.time()
grid_search_t1.fit(X_train, train_labels[0:2000, 1:11])
toc = time.time()
print("grid_search Task 1 finisched | Duration = ", toc-tic, "seconds")

print('!!!!!!!!!!!!! Best Params for Task1 !!!!!!!!!!!!!!!!!!!')
print(grid_search_t1.best_params_)
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

# %% TASK 2:

"""
# Put everything into pipeline
pipe_t2 = Pipeline([('imputer', knn_imputer),
                    ('scaler' , stand_scaler),
                    ('clf'    , clf_t2)],
                    memory=cachedir)
"""

# Put everything into pipeline
pipe_t2 = Pipeline([#('scaler' , stand_scaler),
                    ('clf'    , clf_t2)],
                    memory=cachedir)
"""  
# Create grid with all hperparameters
param_grid_t2 = dict(imputer    = [KNNImputer(), SimpleImputer()],
                     clf__C     = [0.01, 0.1, 1, 10, 100, 1000], 
                     clf__gamma = ['scale', 0.01, 0.1, 1, 10, 1000])
"""

# Create grid with all hperparameters
param_grid_t2 = dict(clf__C     = [0.001, 0.01, 0.1, 1, 10, 100, 1000], 
                     clf__gamma = ['scale', 0.01, 0.1, 1, 10, 1000])

grid_search_t2 = GridSearchCV(pipe_t2, param_grid_t2, scoring='balanced_accuracy')

print("grid_search Task2 started")
tic = time.time()
grid_search_t2.fit(X_train, train_labels[0:2000, 11])
toc = time.time()
print("grid_search Task2 finisched | Duration = ", toc-tic, "seconds")

print('!!!!!!!!!!!!! Best Params for Task2 !!!!!!!!!!!!!!!!!!!')
print(grid_search_t2.best_params_)
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

# %% TASK 3
"""
# Put everything into pipeline
pipe_t3 = Pipeline([('imputer', knn_imputer),
                 ('scaler', stand_scaler),
                 ('regressor', ridge_regressor)],
                memory=cachedir)
"""

# Put everything into pipeline
pipe_t3 = Pipeline([#('scaler', stand_scaler),
                 ('regressor', ridge_regressor)],
                memory=cachedir)

# Create grid with all hperparameters
param_grid_t3 = dict(regressor__alpha=[0.1, 1, 10, 100, 1000, 2000])

# Make grid search for best paramters with 5-fold CrossVal
grid_search_t3 = GridSearchCV(pipe_t3, param_grid_t3, scoring='r2')

print("grid_search X_train_v2 started")
tic = time.time()
grid_search_t3.fit(X_train, train_labels[0:2000, 12:])
toc = time.time()
print("grid_search  finisched | Duration = ", toc-tic, "seconds")

print('!!!!!!!!!!!!! Best Params for Task3 !!!!!!!!!!!!!!!!!!!')
print(grid_search_t3.best_params_)
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

# %%
# Clear the cache directory when you don't need it anymore
rmtree(cachedir)

# %% Refit models with all the trainings samples and the best parameters 

X_train_v2 = train_features_vector_v2
X_train    = train_features_vector

# Task 1
print("Refit Task 1 started")
tic = time.time()
# grid_search_t1.fit(X_train_v2, train_labels[:, 1:11])
pipe_t1_final = Pipeline([#('scaler' , stand_scaler),
                          ('clf'    , OneVsRestClassifier(SVC(C=100, gamma=0.1, class_weight='balanced'), n_jobs=2))],
                           memory=cachedir)
pipe_t1_final.fit(X_train, train_labels[:, 1:11])
toc = time.time()
print("Refit Task 1  finisched | Duration = ", toc-tic, "seconds")


# Task 2
print("Refit Task 2 started")
tic = time.time()
# grid_search_t2.fit(X_train_v2, train_labels[:, 1:11], clf__C=0.1, clf__gamma=0.01)
pipe_t2_final = Pipeline([#('scaler' , stand_scaler),
                          ('clf'    , SVC(class_weight='balanced', C=0.1, gamma='scale'))],
                         memory=cachedir)
pipe_t2_final.fit(X_train, train_labels[:, 11])
toc = time.time()
print("Refit Task 2  finisched | Duration = ", toc-tic, "seconds")
 
# Task 3
print("Refit Task 3 started")
tic = time.time()
# grid_search_t2.fit(X_train_v2, train_labels[:, 1:11], clf__C=0.1, clf__gamma=0.01)
pipe_t3_final = Pipeline([#('scaler' , stand_scaler),
                          ('regressor', Ridge(alpha=100, fit_intercept=False))],
                         memory=cachedir)
pipe_t3_final.fit(X_train, train_labels[:, 12:])
toc = time.time()
print("Refit Task 3  finisched | Duration = ", toc-tic, "seconds")

# %% Predict labels with best estimator 

# Task1:
print("Predict Task 1 started")
tic = time.time()
# predict_labels = grid_search_t1.predict(X_test)
predict_labels = pipe_t1_final.predict(X_test)
toc = time.time()
print("Predict Task 1  finisched | Duration = ", toc-tic, "seconds")
# predict_distance = grid_search_t1.decision_function(X_test)
predict_distance = pipe_t1_final.decision_function(X_test)
predict_confidence = sigmoid(predict_distance)

# Task2:
print("Predict Task 2 started")
tic = time.time()
# predict_labels_sepsis = grid_search_t2.predict(X_test)
predict_labels_sepsis = pipe_t2_final.predict(X_test)
toc = time.time()
print("Predict Task 2  finisched | Duration = ", toc-tic, "seconds")
# predict_distance_sepsis = grid_search_t2.decision_function(X_test)
predict_distance_sepsis = pipe_t2_final.decision_function(X_test)
predict_confidence_sepsis = sigmoid(predict_distance_sepsis)

# Task3:
print("Predict Task 3 started")
tic = time.time()
# predict_reg = grid_search_t3.predict(X_test)
predict_reg = pipe_t3_final.predict(X_test)
toc = time.time()
print("Predict Task 3 finisched | Duration = ", toc-tic, "seconds")

# %% Save results 

test_pid_array = np.reshape(test_pid, (test_pid.shape[0],1))
predict_confidence_sepsis_array = np.reshape(predict_confidence_sepsis, (predict_confidence_sepsis.shape[0],1))

solution = np.concatenate((predict_confidence, predict_confidence_sepsis_array), axis=1)
solution = np.concatenate((solution, predict_reg), axis=1)
solution = np.concatenate((test_pid_array, solution), axis=1)

solution_df = pd.DataFrame(data   = solution,
                           index  = np.arange(0,solution.shape[0]),
                           columns = train_labels_df.columns)


solution_df.to_csv('prediction.zip', index=False, float_format='%.3f', compression='zip') 

#KNN_Imputer = KNNImputer()
#X_imputed = KNN_Imputer.fit_transform(X_train_v2)

"""
# Fit with first feature vector
print("Fit X_train started")
tic = time.time()
reg.fit(X_train, train_labels)
toc = time.time()
print("Fit X_train  finisched | Duration = ", toc-tic, "seconds")
print("score X_train started")
tic = time.time()
score = reg.score(X_train, train_labels)
toc = time.time()
print("Score finisched | Duration = ", toc-tic, "seconds")

# Fit with second feature vector
print("Fit X_train_v2 started")
tic = time.time()
pipe.fit(X_train_v2, train_labels)
toc = time.time()
print("Fit X_train_v2  finisched | Duration = ", toc-tic, "seconds")
print("score X_train_v2 started")
tic = time.time()
score_v2 = pipe.score(X_train_v2, train_labels)
toc = time.time()
print("Score finisched | Duration = ", toc-tic, "seconds")

"""
