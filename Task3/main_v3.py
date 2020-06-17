
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
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem
# Model Selection
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score 
# Classifiers 
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier
# Metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
# Plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



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
    X_train_active_rep = np.repeat(X_train_active, 10, axis=0)
    
    X_keep = np.append(X_keep, X_train_active_rep, axis=0)
    y_keep = np.append(y_keep, np.ones((X_train_active_rep.shape[0],)))
    
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
ord_enc = OrdinalEncoder()
# fit and transform to train features
train_features_ord_enc = ord_enc.fit_transform(train_features_split)
train_features_enc     = enc.fit_transform(train_features_split).toarray()
# transform test features 
test_features_ord_enc = ord_enc.transform(test_features_split)
test_features_enc     = enc.transform(test_features_split).toarray()

    
#%% Stratified K Fold
# Split data set into train and validation set
print("Start StratifiedKFold")
tic = time.time()
n_splits=3
kf = StratifiedKFold(n_splits=n_splits)
X = train_features_enc
y = train_labels

# Try out different classifiers
list_of_classifiers = [GaussianNB(),
                       LinearSVC(max_iter=10000),
                       DecisionTreeClassifier(random_state=42),
                       RandomForestClassifier(bootstrap = False, random_state=42, class_weight={0:1/15,1:10})]

# class_weight={0:0.5,1:20}
# Allocate Scores 
scores_avg = np.zeros((len(list_of_classifiers),4))
clf_keeper = []
i = 0
for clf in list_of_classifiers:
    
    for train_index, test_index in kf.split(X,y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        
        # TODO: Resample the training subset only
        #X_sampled,y_sampled = resample(X_train, y_train)
        #print(np.round(np.count_nonzero(y_sampled)/len(y_sampled)*100, 2), "% of proteins are active in sampled")
        
        X_sampled, y_sampled = X_train, y_train
        
        clf.fit(X_sampled,y_sampled)
        y_predict = clf.predict(X_test)
        
        # --classification report --
        # print(classification_report(y_test, y_predict, labels=[0,1]))
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_predict, labels=[0,1], average='binary')
        rocs = roc_auc_score(y_test, y_predict)
        
        scores = [precision, recall, f1, rocs]
        
        scores_avg[i,:] = scores_avg[i,:] + np.asarray(scores)/n_splits
        
    clf_keeper.append(clf)
    i = i+1
    
# Easier to read from the 'Variablenmanager'
scores_df = pd.DataFrame(data=scores_avg,    # values
                         index=['GaussNB','LinSVC','DecisionTree','RandomForest'],    # 1st column as index
                         columns=['precision','recall','f1','roc_auc'])  # 1st row as the column names

# Select the classifier with the highest f1 score
clf_best = clf_keeper[np.argmax(scores_df.f1)]

# Refit best model with whole data set 
#X_sampled, y_sampled = resample(train_features_enc, train_labels)
X_sampled, y_sampled = train_features_enc, train_labels
clf_best.fit(X_sampled, y_sampled)

toc = time.time()
print("StratifiedKFold done | Duration = ", toc-tic, "seconds")

#%% Resampling to fix the imbalance issue (Only for training dataset)

# Downsample (Remove some of the majority data points) (randomly?)
# train_features_down = train_features_enc
# train_labels_down   = 

# Upsample (Duplicate some minority datapoints with additional small noise)

# 1st approach: Downsample
# 2nd approach: Upsample
# 3rd approach: Change Metric


# Based on this link:
# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
# Use different classifiers




#%% Plot
# Only works with the Ordinal Encoding

# fig = plt.figure()

# for i in range(0,20):
#     ax = fig.add_subplot(4, 5, 1+i, projection='3d')
    
#     plt.title('First Letter ' + ord_enc.categories_[0][i])
    
#     # Get the 3dimensional fetures when the first letter is fixed (reduces from 4d to 3d)
#     first_letter_feature = train_features_ord_enc[np.where(train_features_ord_enc[:,0] == i)]
#     first_letter_label   = train_labels[np.where(train_features_ord_enc[:,0] == i)]
    
#     ax.scatter3D(first_letter_feature[:,1], first_letter_feature[:,2], first_letter_feature[:,3], c=first_letter_label)

# plt.show()

"""
clf = clf_best

clf = RandomForestClassifier(bootstrap = False, random_state=42, class_weight={0:1/15,1:10})
#X_sampled, y_sampled = resample(train_features_enc, train_labels)
X_sampled, y_sampled = train_features_enc, train_labels
clf.fit(X_sampled, y_sampled)
test_labels_v1 = clf.predict(test_features_enc)
"""

# %% Predict labels of test features with best model
print("start predict")
tic = time.time()
test_labels = clf_best.predict(test_features_enc)
toc = time.time()
print("predict done | Duration = ", toc-tic, "seconds")


# %% Save test labels to csv
#np.savetxt('submission_v3.csv', test_labels, fmt='%1.0f')

