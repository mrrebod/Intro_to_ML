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
# Preprocessing
from sklearn.preprocessing import OneHotEncoder
# Classifiers 
from sklearn.svm import LinearSVC
from sklearn.svm import SVC



# %% Read in data sets and convert to numpy arrays 

# Read the datasets as DataFrames
test_features_df = pd.read_csv('test.csv')  
train_set_df     = pd.read_csv('train.csv')  

# Convert to numpy
test_features  = test_features_df.to_numpy()
train_set      = train_set_df.to_numpy() 
train_features = train_set[:,0]
train_labels   = train_set[:,1]
train_labels = train_labels.astype(int)

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


# %% Train classifier 

#clf = LinearSVC(C = 1000, class_weight = 'balanced', max_iter=10000)
clf = SVC(class_weight = 'balanced')

print("start train")
tic = time.time()
clf.fit(train_features_enc[:,:], train_labels[:])
toc = time.time()
print("training done | Duration = ", toc-tic, "seconds")

# %% Predict labels of test features 
print("start predict")
tic = time.time()
test_labels = clf.predict(test_features_enc)
toc = time.time()
print("predict done | Duration = ", toc-tic, "seconds")

# %% Save test labels to csv
np.savetxt('submission.csv', test_labels, fmt='%1.0f')

