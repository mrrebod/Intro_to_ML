"""
Introduction to Machine Learning: Task1b

Vukasin Lalic & Marco Dober aka Snorlax
"""

import numpy as np    
from sklearn.model_selection import KFold 
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDRegressor

# Read in complete train set from csv file
train_set = np.genfromtxt('train.csv', delimiter=',', skip_header=1)
Id_train  = train_set[:, 0]       # Extract ID from train set
y         = train_set[:, 1]       # Extract y label from train set
X         = train_set[:, 2:]      # Extract feature from train set

# Standardize Data
# initialize scaler 
scaler = StandardScaler()
# applay standardization
#X_stand = scaler.fit_transform(X)
X_stand = X

# compute Phi (with standardized features )
Phi_train_stand = np.zeros((np.shape(X)[0], 21))
Phi_train_stand[:, 0:5]   = X_stand                       # Phi 0-4
Phi_train_stand[:, 5:10]  = np.square(X_stand)            # Phi 5-9
Phi_train_stand[:, 10:15] = np.exp(X_stand)               # Phi 10-14
Phi_train_stand[:, 15:20] = np.cos(X_stand)               # Phi 15-19
Phi_train_stand[:, 20]    = np.ones(np.shape(X_stand)[0]) # Phi 21

# Alpha for cross validation
#alpha_cv = np.linspace(start=1e-3, stop=100, num=100)
alpha_cv = np.logspace(start=-2, stop=2, num=100)

# Split data set into train and test set
kf = KFold(n_splits=10)

# Allocate RMSE vector
#rmse_avg = np.zeros(np.shape(alpha_cv))
rmse_avg = np.zeros([len(alpha_cv),5])

for i in range(len(alpha_cv)):              # For each alpha
    clf_ridge = Ridge(alpha=alpha_cv[i], 
                      fit_intercept = False)
    
    clf_linrg = LinearRegression(
                                 fit_intercept = False)
    
    clf_sgdrg = SGDRegressor(alpha=alpha_cv[i], 
                             max_iter=100000,
                             tol=0.000001,
                             fit_intercept = False,
                             random_state=0)
    
    clf_lasso = Lasso(alpha=alpha_cv[i],
                      max_iter=100000,
                      tol=0.000001,
                      fit_intercept = False)
    
    clf_lisvr = LinearSVR(
                          fit_intercept = False)
    
    for train_index, test_index in kf.split(Phi_train_stand):
        X_train, X_test = Phi_train_stand[train_index], Phi_train_stand[test_index]
        y_train, y_test = y[train_index], y[test_index]
                
#        clf.fit(X_train,y_train)
#        y_predict = clf.predict(X_test)

#        rmse_batch = np.sqrt(np.mean((y_test-y_predict)**2))
#        
#        rmse_avg[i] = rmse_avg[i] + (rmse_batch)/10
        
        clf_ridge.fit(X_train,y_train)
        clf_linrg.fit(X_train,y_train)
        clf_sgdrg.fit(X_train,y_train)
        clf_lasso.fit(X_train,y_train)
        clf_lisvr.fit(X_train,y_train)
        
        y_predict_ridge = clf_ridge.predict(X_test)
        y_predict_linrg = clf_linrg.predict(X_test)
        y_predict_sgdrg = clf_sgdrg.predict(X_test)
        y_predict_lasso = clf_lasso.predict(X_test)
        y_predict_lisvr = clf_lisvr.predict(X_test)
        
        y_predict = [y_predict_ridge,
                     y_predict_linrg,
                     y_predict_sgdrg,
                     y_predict_lasso,
                     y_predict_lisvr]
        
        for j in range(len(y_predict)):
        
            rmse_batch = np.sqrt(np.mean((y_test-y_predict[j])**2))
            
            rmse_avg[i][j] = rmse_avg[i][j] + (rmse_batch)/10
        
# Where is the rmse the smallest
np.argmin(rmse_avg,axis=0)

clf_ridge = Ridge(alpha=alpha_cv[np.argmin(rmse_avg,axis=0)[0]], 
                  fit_intercept = False)

clf_linrg = LinearRegression(
                             fit_intercept = False)

clf_sgdrg = SGDRegressor(alpha=alpha_cv[np.argmin(rmse_avg,axis=0)[2]], 
                         max_iter=100000,
                         tol=0.000001,
                         fit_intercept = False,
                         random_state=0)

clf_lasso = Lasso(alpha=alpha_cv[np.argmin(rmse_avg,axis=0)[3]],
                  max_iter=100000,
                  tol=0.000001,
                  fit_intercept = False)

clf_lisvr = LinearSVR(
                      fit_intercept = False)


clf_ridge.fit(Phi_train_stand, y)
clf_linrg.fit(Phi_train_stand, y)
clf_sgdrg.fit(Phi_train_stand, y)
clf_lasso.fit(Phi_train_stand, y)
clf_lisvr.fit(Phi_train_stand, y)

# save coefficients into array
weights = np.zeros([21,5])
weights[:,0] = clf_ridge.coef_
weights[:,1] = clf_linrg.coef_
weights[:,2] = clf_sgdrg.coef_
weights[:,3] = clf_lasso.coef_
weights[:,4] = clf_lisvr.coef_

# Save Data to csv file      
int(np.median(np.argmin(rmse_avg,axis=1))) # this gives the model with the smallest rmse overall
np.savetxt('submission.csv',weights[:,int(np.median(np.argmin(rmse_avg,axis=1)))])