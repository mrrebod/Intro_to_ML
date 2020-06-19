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
from sklearn.neural_network import MLPClassifier


#%% Resample
def resample(X_train, y_train, sampling_rate=15, repetition=5, upsamp=0):
    
    # sampling_rate = 15
    # Step 1: Downsample (Remove some of the majority data points)
    numbers_to_keep = np.count_nonzero(y_train) * sampling_rate # Arbitrarly choosen
    X_train_inactive = X_train[y_train==0]
    X_train_active   = X_train[y_train==1]
    
    print(X_train_active.shape, "Shape of X_train_active")
    print(X_train_inactive.shape, "Shape of X_train_inactive")
    
    if (numbers_to_keep > X_train_inactive.shape[0]):
        numbers_to_keep = X_train_inactive.shape[0] # Limits maximum
    
    index_to_keep = np.random.choice(X_train_inactive.shape[0], numbers_to_keep, replace=False) 
    
    # Put the resulting array together
    X_keep = X_train_inactive[index_to_keep]
    y_keep = np.zeros((X_keep.shape[0],))


    #Step 2: Add the active features again
    X_keep = np.append(X_keep, X_train_active, axis=0)
    y_keep = np.append(y_keep, np.ones((X_train_active.shape[0],)))
    
    
    # Step 3: Replicate the active features to give them more weight (optional)
    X_train_active_rep = np.repeat(X_train_active, repetition, axis=0)
    
    X_keep = np.append(X_keep, X_train_active_rep, axis=0)
    y_keep = np.append(y_keep, np.ones((X_train_active_rep.shape[0],)))
    
    
    #Step 4: Upsample (add closest neighbours as active features)
    if upsamp == 1:
        all_neighbour_index = []
        for ii in range(0,X_train_active.shape[0]):
            
            distance = (X_train_active[ii,:]^test_features_enc).sum(axis=1)/2
            
            # Only use the distance where it is exactly = 1 as a neighbour
            neigbour_index = np.where(distance == 1)
            
            # Save it
            all_neighbour_index = np.append(all_neighbour_index,neigbour_index)
            
        # Extract the neighbours
        unique_neighbour_index = np.unique(all_neighbour_index)
        unique_neighbour_index = unique_neighbour_index.astype(int)
        X_neighbours = test_features_enc[unique_neighbour_index,:]
        
        print(X_neighbours.shape, "Shape of X_neighbours") # size = (17000,80)
        
        # Only use every 10th row, otherwise too big
        X_neighbours = X_neighbours[::20,:]
        print(X_neighbours.shape, "Shape of X_neighbours after reduction")
        
        
        # Only check possible neighbours in the test feature vector, because the
        # rest has been already set in the train fetaures vector
        
        X_keep = np.append(X_keep, X_neighbours, axis=0)
        y_keep = np.append(y_keep, np.ones((X_neighbours.shape[0],)))    
        
    
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

np.random.seed(694201312)

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
enc = OneHotEncoder(handle_unknown='ignore',dtype=int)
# fit and transform to train features
train_features_enc     = enc.fit_transform(train_features_split).toarray()
# transform test features 
test_features_enc      = enc.transform(test_features_split).toarray()

    
#%% Stratified K Fold
# Split data set into train and validation set
print("Start Fitting")
tic = time.time()

clf_best = MLPClassifier(max_iter=200, alpha=0.00001, random_state=0)

# Fit best model with whole data set 
X_sampled, y_sampled = resample(train_features_enc, train_labels, sampling_rate=30, repetition=15, upsamp=0)
clf_best.fit(X_sampled, y_sampled)

toc = time.time()
print("Fitting done | Duration = ", toc-tic, "seconds")


# %% Predict labels of test features with best model
print("start predict")
tic = time.time()
test_labels = clf_best.predict(test_features_enc)
toc = time.time()
print("predict done | Duration = ", toc-tic, "seconds")


# %% Save test labels to csv
np.savetxt('submission_best.csv', test_labels, fmt='%1.0f')

