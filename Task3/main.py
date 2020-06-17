# -*- coding: utf-8 -*-
"""
Introduction to Machine Learning: Task3

Marco Dober & Vukasin Lalic aka Snorlax
"""

# %% Import libraries 

# General stuff 
import numpy as np 
import pandas as pd
import time
from sklearn.utils import shuffle
# Preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.feature_selection import SelectKBest
# Model Selection
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
# Classifiers 
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier


#%% Resample
def resample(X_train, y_train):
    
    sampling_rate = 15
    # Step 1: Downsample (Remove some of the majority data points)
    numbers_to_keep = np.count_nonzero(y_train) * sampling_rate # Arbitrarly choosen
    X_train_inactive = X_train[y_train==0]
    X_train_active   = X_train[y_train==1]
    
    index_to_keep = np.random.choice(X_train_inactive.shape[0], numbers_to_keep, replace=False) 
    
    # Put the resulting array together
    X_keep = X_train_inactive[index_to_keep]
    y_keep = np.zeros((X_keep.shape[0],))

    
    # Step 2: Upsample (Duplicate some minority datapoints with additional small noise)
    # numbers_to_add
    X_train_active_rep = np.repeat(X_train_active, 15, axis=0)
    
    X_keep = np.append(X_keep, X_train_active_rep, axis=0)
    y_keep = np.append(y_keep, np.ones((X_train_active_rep.shape[0],)))
    
    X_keep, y_keep = shuffle(X_keep, y_keep, random_state=42)
    
    return X_keep, y_keep

# %% Read in data sets and convert to numpy arrays 

# Read the datasets as DataFrames
test_features_df = pd.read_csv('test.csv')  
train_set_df     = pd.read_csv('train.csv')  

# Convert to numpy
test_features  = test_features_df.to_numpy()
test_features  = test_features.flatten()
train_set      = train_set_df.to_numpy() 
train_features = train_set[:,0]
train_labels   = train_set[:,1]
train_labels   = train_labels.astype(int)

# %% Statistics of train set

# How unbalanced is train set? 
# Count number of actives in train labels
numb_of_actives   = np.count_nonzero(train_labels == 1)
# count number of inactives in train labels 
numb_of_inactives = np.count_nonzero(train_labels == 0)
# Check if there are non-binary values in train labels 
if numb_of_actives + numb_of_inactives != len(train_labels):
    print('!!! ERROR !!! : There are non-binary values in train_labels')
# Calculate and display % of acitve proteins in train labels 
perc_of_actives = np.round(numb_of_actives/len(train_labels)*100, 2)
print(perc_of_actives, "% of proteins are active in train labels")

# Do some acid combinations occure multiple times ? 
train_features_unique = np.unique(train_features)
if len(train_features) != len(train_features_unique):
    print('!!! ATTENTION !!! : Some acid combinations occure miltiple times')
else:
    del train_features_unique

# %% Encoding features 
# We need to encode the features since sklearn can only handle numerical features
# One-hot-encoding is the most used method 

# Split the feature ['ABCD']  to ['A', 'B', 'C', 'D']

# Split train features 
train_features_split = np.zeros([len(train_features), 4], dtype=str)
for i in range(0,len(train_features)):
    train_features_split[i,:] = list(train_features[i])
    
# Split test features 
test_features_split = np.zeros([len(test_features), 4], dtype=str)
for i in range(0,len(test_features)):
    test_features_split[i,:] = list(test_features[i])
    
# One-hot-encode train and test features 
# initalize type of encoder 
enc = OneHotEncoder(handle_unknown='ignore')
# fit and transform to train features
train_features_enc = enc.fit_transform(train_features_split).toarray()
# transform test features 
test_features_enc = enc.transform(test_features_split).toarray()


# %% Train Classifier

# Components of pipeline

# Scaler 
stand_scaler = StandardScaler()

# Feature selection 
select = SelectKBest()

# Classifier 
clf_RandomForest = RandomForestClassifier(bootstrap = False, random_state=42)
clf_AdaBoost = AdaBoostClassifier(random_state=42)
    
# Create pipeline
pipe = Pipeline([#('scaler', stand_scaler),
                 #('selector', select),
                 ('clf', clf_RandomForest)]) 

# Hyperparameters to evaluate best model 
param_grid = dict(#scaler            = ['passthrough', stand_scaler],
                  #selector__k       = ['all', 10, 30, 60],
                  clf               = [clf_AdaBoost],
                  #clf__class_weight = [{0:0.05,1:25}, {0:0.03,1:30}],
                  clf__n_estimators = [300],
                 #clf__max_features = ['sqrt', None]
                  )

# Make grid search for best model
grid_search = GridSearchCV(pipe, 
                           param_grid, 
                           scoring=['f1', 'precision', 'recall'], 
                           cv=StratifiedKFold(n_splits=3), 
                           refit='f1')


train_features_enc, train_labels = resample(train_features_enc, train_labels)

# Fit to train data 
print("grid_search started")
tic = time.time()
grid_search.fit(train_features_enc, train_labels)
toc = time.time()
print("grid_search  finisched | Duration = ", toc-tic, "seconds")

print('!!!!!!!!!!!!! Best Params !!!!!!!!!!!!!!!!!!!')
print(grid_search.best_params_)
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
print('!!!!!!!!!!!!! Best CV Score !!!!!!!!!!!!!!!!!!!')
print(grid_search.best_score_)
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

# Save results into panda frame
cv_result_df = pd.DataFrame(grid_search.cv_results_)

# %% Predict labels of test features with best model
print("start predict")
tic = time.time()
test_labels = grid_search.predict(test_features_enc)
toc = time.time()
print("predict done | Duration = ", toc-tic, "seconds")

# %% Save test labels to csv
#np.savetxt('submission.csv', test_labels, fmt='%1.0f')

