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

from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input, decode_predictions

# Metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score

# Classifiers 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

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
feat_size = 512 # depends on used model (VGG16: 512) (Xeption: 2048)
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

    img_path = ''.join(['food/', str(img_nr).zfill(5),'.jpg'])
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
    # only print every 5%, otherwise too fast & a printing error occurs
    if (trp_nr%round(nr_of_train_triplets*0.05) == 0):
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
    # only print every 5%, otherwise too fast & a printing error occurs
    if (trp_nr%round(nr_of_test_triplets*0.05) == 0):
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

print("Start Classification")
tic = time.time()

X_train = trp_train_features
y_train = trp_train_features_labels
X_test  = trp_test_features

# clf = RandomForestClassifier(n_estimators=300, random_state=0)
clf = MLPClassifier(random_state=0,alpha=0.00005, max_iter=700)
# clf = LinearSVC(random_state=0, C=100)
# clf = GaussianProcessClassifier(random_state=0)
# clf = GaussianNB()
# clf = QuadraticDiscriminantAnalysis()

sub_tic = time.time()
print("Fitting started")
clf.fit(X_train, y_train)
sub_toc = time.time()
print("Duration: ", sub_toc-sub_tic, "seconds")

sub_tic = time.time()
print("Predicting started")
y_predict = clf.predict(X_test)
sub_toc = time.time()
print("Duration: ", sub_toc-sub_tic, "seconds")

toc = time.time()
print("Classification done | Duration = ", toc-tic, "seconds")

# %% Save test labels to csv
np.savetxt('submission.csv', y_predict, fmt='%1.0f')
print("Saved")
