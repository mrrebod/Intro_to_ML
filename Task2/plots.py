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
from sklearn.linear_model import LinearRegression
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.preprocessing import PolynomialFeatures

# -----------------------------------------------------------------------------

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# %%-----------------------------------------------------------------------------

# Read the datasets as DataFrames
test_features_df  = pd.read_csv('test_features.csv')  
train_features_df = pd.read_csv('train_features.csv')  
train_labels_df   = pd.read_csv('train_labels.csv')  

# Convert to numpy
test_features  = test_features_df.to_numpy()
train_features = train_features_df.to_numpy() 
train_labels   = train_labels_df.to_numpy()

# %%-----------------------------------------------------------------------------
# Calculate the global mean of each column (except pid)
# global_mean   = np.nanmean(train_features[:,1:], axis=0)
global_mean = np.nanmedian(train_features[:,1:], axis=0)
      
# %%-----------------------------------------------------------------------------
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




# %%-----------------------------------------------------------------------------
# Plot some data aginst age and time




# %%-----------------------------------------------------------------------------
# How often complete 12h-colums are nan 

occurance_of_nan = np.zeros(patients_data.shape[1])

for i in range(patients_data.shape[2]):
    nans_per_col = np.count_nonzero(np.isnan(patients_orig[:,:,i]), axis=0)
    which_cols_are_nan = np.where(nans_per_col == 12)
    
    np.add.at(occurance_of_nan, which_cols_are_nan, 1) # np.ufunc.at()

# Convert to percentages
occurance_of_nan = occurance_of_nan/patients_data.shape[2]*100

mask1 = occurance_of_nan < 90
mask2 = occurance_of_nan >= 90
mask3 = occurance_of_nan >= 95
mask4 = occurance_of_nan <= 10

colums_to_delete = np.where(mask2) # delete all cols with more than 90% missing
colums_to_keep   = np.where(mask4) # keep only cols with less than 10% missing


# Plot
plt.figure()
plt.bar(np.arange(patients_data.shape[1])[mask1], occurance_of_nan[mask1], color = 'blue')
plt.bar(np.arange(patients_data.shape[1])[mask2], occurance_of_nan[mask2], color = 'red')
plt.bar(np.arange(patients_data.shape[1])[mask3], occurance_of_nan[mask3], color = 'darkred')
plt.ylabel("Percentage [%]")
plt.xticks(np.arange(patients_data.shape[1]), train_features_df.columns[1:], rotation=90)
plt.grid()

legend_elements = [patches.Rectangle( xy=(0,0), width=0, height=0, facecolor='blue', label='< 90% nan' ),
                   patches.Rectangle( xy=(0,0), width=0, height=0, facecolor='red', label='> 90% nan' ),
                   patches.Rectangle( xy=(0,0), width=0, height=0, facecolor='darkred', label='> 95% nan' )]
plt.legend(handles=legend_elements)


# %%-----------------------------------------------------------------------------
# Use a poly fit of order 2 or 3 to fit data if more than 2 points not nan

# 2, 6, 21, 24, 34, 
# p_d_test = patients_data[:,2,3]

regr = LinearRegression()
regr_list = np.zeros((2, patients_data.shape[1])) # average the fit for each feature
pid = 3

n = 2
p_d_test = patients_data[:, n, pid]
x_axis   = patients_data[:, 0, pid] # Time as x-axis

nan_index     = np.argwhere( np.isnan(patients_data[:, n, pid]))
non_nan_index = np.argwhere(~np.isnan(patients_data[:, n, pid]))

p_d_test_no_nan = patients_data[1:, n, pid]
x_axis_no_nan   = patients_data[1:, 0, pid]


regr.fit(x_axis_no_nan.reshape(-1, 1), p_d_test_no_nan)

regr_list[0, n] = regr.coef_
regr_list[1, n] = regr.intercept_

print("Coeffs:    ", regr.coef_)
print("Intercept: ", regr.intercept_)

x_axis_nan = np.array([1])
p_d_test_nan = regr.predict(x_axis_nan.reshape(-1, 1))

plt.figure()
plt.plot(x_axis_no_nan, p_d_test_no_nan, 'bo')
plt.plot(x_axis_nan, p_d_test_nan, 'ro')
plt.plot(x_axis_nan, x_axis_nan*regr.coef_ + regr.intercept_, 'gx')
plt.plot(x_axis, x_axis*regr.coef_ + regr.intercept_)



