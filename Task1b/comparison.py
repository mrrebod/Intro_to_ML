"""
Introduction to Machine Learning: Task1b

Vukasin Lalic & Marco Dober aka Snorlax
"""

import numpy as np    
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Lasso
from sklearn.svm import LinearSVR

# Read in complete train set from csv file
train_set = np.genfromtxt('train.csv', delimiter=',', skip_header=1)
Id_train  = train_set[:, 0]       # Extract ID from train set
y         = train_set[:, 1]       # Extract y label from train set
X         = train_set[:, 2:]      # Extract feature from train set

# No Scaling
X_stand = X

# compute Phi (with standardized features )
Phi_train_stand = np.zeros((np.shape(X)[0], 21))
Phi_train_stand[:, 0:5]   = X_stand                       # Phi 0-4
Phi_train_stand[:, 5:10]  = np.square(X_stand)            # Phi 5-9
Phi_train_stand[:, 10:15] = np.exp(X_stand)               # Phi 10-14
Phi_train_stand[:, 15:20] = np.cos(X_stand)               # Phi 15-19
Phi_train_stand[:, 20]    = np.ones(np.shape(X_stand)[0]) # Phi 21

# Alpha for cross validation
alpha_cv = np.logspace(start = -3, stop = 4, num = 100)

# Regularization Parameter for Support Vector Machine
C_cv = 1/alpha_cv

# Split data set into train and test set
kf = KFold(n_splits = 10,
           shuffle = True,
           random_state = 0)

# Allocate RMSE and weights vector
rmse_avg = np.zeros([len(alpha_cv),5])
weights_avg = np.zeros([21,5])


for i in range(len(alpha_cv)):              # For each alpha
    clf_ridge = Ridge(alpha = alpha_cv[i], 
                      fit_intercept = False)
    
    clf_linrg = LinearRegression(fit_intercept = False)
    
    clf_sgdrg = SGDRegressor(alpha = alpha_cv[i], 
                             max_iter = 100000,
                             tol = 0.0001,
                             fit_intercept = False,
                             random_state = 1)
    
    clf_lasso = Lasso(alpha = alpha_cv[i],
                      max_iter = 100000,
                      tol = 0.0001,
                      fit_intercept = False)
    
    clf_lisvr = LinearSVR(C = C_cv[i],
                          max_iter = 1000000,
                          random_state = 0,
                          fit_intercept = False)
    
    for train_index, test_index in kf.split(Phi_train_stand):
        X_train, X_test = Phi_train_stand[train_index], Phi_train_stand[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        
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
            
            rmse_avg[i][j] = rmse_avg[i][j] + (rmse_batch)/kf.n_splits
        
# Where is the rmse the smallest
np.argmin(rmse_avg,axis=0)
min_model = np.argmin(np.min(rmse_avg,axis=0))
print("Min Model: ", min_model)

# Select Best method
if min_model == 0:
    clf_best = Ridge(alpha = alpha_cv[np.argmin(rmse_avg,axis=0)[0]], 
                     fit_intercept = False)
    print("Best Method is Ridge with alpha = ", alpha_cv[np.argmin(rmse_avg,axis=0)[0]])
    
elif min_model == 1:
    clf_best = LinearRegression(fit_intercept = False)
    print("Best Method is LinearRegression")
    
elif min_model == 2:
    clf_best = SGDRegressor(alpha = alpha_cv[np.argmin(rmse_avg,axis=0)[2]], 
                            max_iter = 100000,
                            tol = 0.0001,
                            fit_intercept = False,
                            random_state = 1)
    print("Best Method is SGDRegressor with alpha = ", alpha_cv[np.argmin(rmse_avg,axis=0)[2]])

elif min_model == 3:
    clf_best = Lasso(alpha = alpha_cv[np.argmin(rmse_avg,axis=0)[3]],
                     max_iter = 100000,
                     tol = 0.0001,
                     fit_intercept = False)
    print("Best Method is Lasso with alpha = ", alpha_cv[np.argmin(rmse_avg,axis=0)[3]])

else:
    clf_best = LinearSVR(C = C_cv[np.argmin(rmse_avg,axis=0)[4]],
                         max_iter = 1000000,
                         random_state = 0,
                         fit_intercept = False)
    print("Best Method is LinearSVR with C = ", C_cv[np.argmin(rmse_avg,axis=0)[4]])


# This code block was only used to discuss the best soultion
rmse_batch = np.array([])
weights_avg_new = np.zeros(np.shape(21,))
rmse_avg_new = 0
for train_index, test_index in kf.split(Phi_train_stand):
        X_train, X_test = Phi_train_stand[train_index], Phi_train_stand[test_index]
        y_train, y_test = y[train_index], y[test_index]
                
        clf_best.fit(X_train,y_train)
        
        y_predict = clf_best.predict(X_test)

        rmse_batch = np.append(rmse_batch, np.sqrt(np.mean((y_test-y_predict)**2)))
        
        rmse_avg_new = rmse_avg_new + (rmse_batch[-1])/10
        weights_avg_new = weights_avg_new + clf_best.coef_/10
        

# Fit the best model on the whole training set and get it's weights
clf_best.fit(Phi_train_stand, y)

# save coefficients into array
weights = np.zeros([21,1])
weights[:,0] = clf_best.coef_

# Save Data to csv file      
np.savetxt('submission.csv', weights)