# -*- coding: utf-8 -*-
"""
Introduction to Machine Learning: Task4

Marco Dober & Vukasin Lalic aka Snorlax
"""

# %% Import libraries 
# General stuff 
import numpy as np 
import pandas as pd
import time

# Import Keras and pretrained models
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

# Metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# Classifiers 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# Model Selection
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


# %% Fold

# def mySplit(features, labels, n_triplets=100, n_splits=2):
def mySplit(n_triplets=100, n_splits=2):
    # n_triplets = 100
    # n_splits = 2
    
    nt2 = int(n_triplets*(n_splits-1)/n_splits)
    nt3 = int(n_triplets*(1)/n_splits)
    
    
    # n_triplets = 100          # numer of triplets for test and train each
    # nt2  = int(n_triplets/2)  # half of triplets have pos and other half neg features
    # nt3  = int(2*nt2)
    
    # Separate training and validation set so that they don't have common images
    # Get pos features
    X_train      = trp_train_features       [0:nt2,:]
    y_train      = trp_train_features_labels[0:nt2]
    # append neg features
    X_train      = np.append(X_train, trp_train_features       [nr_of_train_triplets:nr_of_train_triplets+nt2,:],axis=0)
    y_train      = np.append(y_train, trp_train_features_labels[nr_of_train_triplets:nr_of_train_triplets+nt2])
    
    img_in_train = train_triplets[0:nt2,:] # pos and neg triplets have same images
    img_in_train = np.unique(img_in_train.flatten()) # these are the image numbers in train
    
    
    # Get pos features
    X_test      = trp_train_features       [nt2:nt3,:]
    y_test      = trp_train_features_labels[nt2:nt3]
    # append neg features
    X_test      = np.append(X_test, trp_train_features       [nr_of_train_triplets+nt2:nr_of_train_triplets+nt3,:],axis=0)
    y_test      = np.append(y_test, trp_train_features_labels[nr_of_train_triplets+nt2:nr_of_train_triplets+nt3])
    
    img_in_test = train_triplets[nt2:nt3,:] # pos and neg triplets have same images
    img_in_test = np.append(img_in_test, train_triplets[nt2:nt3,:],axis=0) # b & c in wrong order, but it doesn't matter since we're only interested in the index of occurance
    
    img_in_test_which_also_occur_in_train = np.isin(img_in_test, img_in_train, assume_unique=False, invert=False)
    
    # These indices contain images from test and train set -> remove them from test
    X_test = np.delete(X_test, np.where(img_in_test_which_also_occur_in_train)[0], 0) # last arg: 0=row, 1=col
    y_test = np.delete(y_test, np.where(img_in_test_which_also_occur_in_train)[0], 0)
    
    return X_train, y_train, X_test, y_test


# %% Read in data sets and convert to numpy arrays 

# Read the datasets as DataFrames
test_triplets_df  = pd.read_csv('test_triplets.txt',  header=None, names=['a','b','c'], delim_whitespace=True)  
train_triplets_df = pd.read_csv('train_triplets.txt', header=None, names=['a','b','c'], delim_whitespace=True)  

# Convert to numpy
test_triplets  = test_triplets_df.to_numpy()
train_triplets = train_triplets_df.to_numpy() 


# %% Feature Generation
print("Start Feature Generation")
tic = time.time()

model = VGG16(weights='imagenet', input_shape=(224,224,3), include_top=False, pooling='avg')
nr_of_train_triplets = 59515 # all = 59515
nr_of_test_triplets  = 59544 # all = 59544
nr_of_images = 10000
feat_size = 512 # depends on used model (VGG16: 512)
#%%
pos_train_features = np.zeros((nr_of_train_triplets, feat_size*3))
neg_train_features = np.zeros((nr_of_train_triplets, feat_size*3))
trp_test_features  = np.zeros((nr_of_test_triplets,  feat_size*3))
img_features       = np.zeros((nr_of_images,         feat_size))


#%% Calculate all image features ------------------------------------------------
sub_tic = time.time()
print("Calculation Image Features")
for img_nr in range(0, nr_of_images):
    print("\rProgress: ",str(img_nr),"/",str(nr_of_images-1), end='\r', flush=True)

    img_path = ''.join(['food/food/', str(img_nr).zfill(5),'.jpg'])
    img = image.load_img(img_path, target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    img_features[img_nr,:] = model.predict(x)
print("") # New line after the progress indicator

sub_toc = time.time()
print("Duration: ", sub_toc-sub_tic, "seconds")
print("")


#%% Calculate all train triplets features -------------------------------------
sub_tic = time.time()
print("Concatenation Training Triplets")
for trp_nr in range(0, nr_of_train_triplets):
    # TODO only print every 5%, otherwise too fast & a printing error occurs
    print("\rProgress: ",str(trp_nr),"/",str(nr_of_train_triplets-1), end='\r', flush=True)

    a_img_nr = train_triplets[trp_nr, 0]
    b_img_nr = train_triplets[trp_nr, 1]
    c_img_nr = train_triplets[trp_nr, 2]
    
    a = img_features[a_img_nr, :]
    b = img_features[b_img_nr, :]
    c = img_features[c_img_nr, :]
    
    pos_train_features[trp_nr,:] = np.concatenate((a,b,c))
    neg_train_features[trp_nr,:] = np.concatenate((a,c,b))

# Feature Vector & Label
trp_train_features        = np.append(pos_train_features, neg_train_features ,axis=0)
trp_train_features_labels = np.append(np.ones((nr_of_train_triplets)), np.zeros((nr_of_train_triplets)))
print("") # New line after the progress indicator

sub_toc = time.time()
print("Duration: ", sub_toc-sub_tic, "seconds")
print("")


#%% Calculate all test triplets features --------------------------------------
sub_tic = time.time()
print("Concatenation Testing Triplets")
for trp_nr in range(0, nr_of_test_triplets):
    # TODO only print every 5%, otherwise too fast & a printing error occurs
    print("\rProgress: ",str(trp_nr),"/",str(nr_of_test_triplets-1), end='\r', flush=True)

    a_img_nr = test_triplets[trp_nr, 0]
    b_img_nr = test_triplets[trp_nr, 1]
    c_img_nr = test_triplets[trp_nr, 2]
    
    a = img_features[a_img_nr, :]
    b = img_features[b_img_nr, :]
    c = img_features[c_img_nr, :]
    
    trp_test_features[trp_nr,:] = np.concatenate((a,b,c))
print("") # New line after the progress indicator

sub_toc = time.time()
print("Duration: ", sub_toc-sub_tic, "seconds")
print("")

toc = time.time()
print("Feature Generation done | Duration = ", toc-tic, "seconds")

# %% Train classifier for prediction


print("Setup Test and Train Set")
tic = time.time()

# n_triplets = 100
# n_splits = 2

# nt2 = int(n_triplets*(n_splits-1)/n_splits)
# nt3 = int(n_triplets*(1)/n_splits)


n_triplets = 5000          # numer of triplets for test and train each
nt2  = int(n_triplets/2)  # half of triplets have pos and other half neg features
nt3  = int(2*nt2)

# Separate training and validation set so that they don't have common images
# Get pos features
X_train      = trp_train_features       [0:nt2,:]
y_train      = trp_train_features_labels[0:nt2]
# append neg features
X_train      = np.append(X_train, trp_train_features       [nr_of_train_triplets:nr_of_train_triplets+nt2,:],axis=0)
y_train      = np.append(y_train, trp_train_features_labels[nr_of_train_triplets:nr_of_train_triplets+nt2])

img_in_train = train_triplets[0:nt2,:] # pos and neg triplets have same images
img_in_train = np.unique(img_in_train.flatten()) # these are the image numbers in train


# Get pos features
X_test      = trp_train_features       [nt2:nt3,:]
y_test      = trp_train_features_labels[nt2:nt3]
# append neg features
X_test      = np.append(X_test, trp_train_features       [nr_of_train_triplets+nt2:nr_of_train_triplets+nt3,:],axis=0)
y_test      = np.append(y_test, trp_train_features_labels[nr_of_train_triplets+nt2:nr_of_train_triplets+nt3])

img_in_test = train_triplets[nt2:nt3,:] # pos and neg triplets have same images
img_in_test = np.append(img_in_test, train_triplets[nt2:nt3,:],axis=0) # b & c in wrong order, but it doesn't matter since we're only interested in the index of occurance

img_in_test_which_also_occur_in_train = np.isin(img_in_test, img_in_train, assume_unique=False, invert=False)

# These indices contain images from test and train set -> remove them from test
X_test = np.delete(X_test, np.where(img_in_test_which_also_occur_in_train)[0], 0) # last arg: 0=row, 1=col
y_test = np.delete(y_test, np.where(img_in_test_which_also_occur_in_train)[0], 0)

toc = time.time()
print("Setup done | Duration = ", toc-tic, "seconds")

print("Start Classification")
tic = time.time()
# n_triplets=100
# n_splits=2

# X_train = trp_train_features
# y_train = trp_train_features_labels
# X_test  = trp_test_features

# clf = RandomForestClassifier(n_estimators=150, random_state=0)
# clf = MLPClassifier(max_iter=200, random_state=0)
# clf = LinearSVC(random_state=0, C=100)


list_of_classifiers = [MLPClassifier(random_state=0,alpha=0.00005, max_iter=500),
                       MLPClassifier(random_state=0,alpha=0.0001, max_iter=500),
                       MLPClassifier(random_state=0,alpha=0.001, max_iter=500),
                       MLPClassifier(random_state=0,alpha=0.01, max_iter=500),
                       MLPClassifier(random_state=0,alpha=0.1, max_iter=500),
                       MLPClassifier(random_state=0,alpha=1, max_iter=500),
                       MLPClassifier(random_state=0,alpha=10, max_iter=500),
                       RandomForestClassifier(n_estimators=150,random_state=0),
                       RandomForestClassifier(n_estimators=300,random_state=0),
                       # QuadraticDiscriminantAnalysis(reg_param=0.0),
                       # QuadraticDiscriminantAnalysis(reg_param=1.0),
                       # QuadraticDiscriminantAnalysis(reg_param=10.0),
                       # QuadraticDiscriminantAnalysis(reg_param=100.0),
                       GaussianNB(),
                       SVC(kernel='rbf', C=1),
                       SVC(kernel='rbf', C=10),
                       SVC(kernel='rbf', C=100),
                       SVC(kernel='rbf', C=1000)]

names = ['alpha:0.00005  ',
         'alpha:0.0001   ',
         'alpha:0.001    ',
         'alpha:0.01     ',
         'alpha:0.1      ',
         'alpha:1        ',
         'alpha:10       ',
         'n_est:150      ',
         'n_est:300      ',
         # 'QDC:0',
         # 'QDC:1',
         # 'QDC:10',
         # 'QDC:100',
         'GaussNB        ',
         'SVC, rbf C:1   ',
         'SVC, rbf C:10  ',
         'SVC, rbf C:100 ',
         'SVC, rbf C:1000']

# QDC: TypeError: array type float16 is unsupported in linalg

scores_avg = np.zeros((len(list_of_classifiers),2))
clf_keeper = []
i = 0


for clf in list_of_classifiers:
    sub_tic = time.time()
    print("Fitting ",names[i],"=> ", end='')
    clf.fit(X_train,y_train)
    y_predict = clf.predict(X_test)
    
    # precision, recall, f1, support = precision_recall_fscore_support(y_test, y_predict, labels=[0,1], average='binary')
    # rocs = roc_auc_score(y_test, y_predict)
    acc = accuracy_score(y_test, y_predict)
    
    sub_toc = time.time()
    print("", sub_toc-sub_tic, "seconds")
        
    scores = [acc, (sub_toc-sub_tic)]
    scores_avg[i,:] = scores
            
    # scores_avg[i,:] = scores_avg[i,:] + np.asarray(scores)/n_splits
    clf_keeper.append(clf)
    i = i+1
    

# Easier to read from the 'Variablenmanager'
scores_df = pd.DataFrame(data=scores_avg,    # values
                          index=names,    # 1st column as index
                          columns=['acc','duration'])  # 1st row as the column names

# Select the classifier with the highest f1 score
clf_best = clf_keeper[np.argmax(scores_df.acc)]

# sub_tic = time.time()
# print("Predicting started")
# y_predict = clf_best.predict(trp_test_features)
# sub_toc = time.time()
# print("Duration: ", sub_toc-sub_tic, "seconds")

toc = time.time()
print("Classification done | Duration = ", toc-tic, "seconds")

# %% Save test labels to csv
np.savetxt('submission_grid.csv', y_predict, fmt='%1.0f')
print("Saved")
