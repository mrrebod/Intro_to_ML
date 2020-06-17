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

# Resize images
from skimage.transform import resize
import matplotlib.image as mpimg

# Metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score

# Classifiers 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

# %% Read in data sets and convert to numpy arrays 

# Read the datasets as DataFrames
test_triplets_df  = pd.read_csv('test_triplets.txt',  header=None, names=['a','b','c'], delim_whitespace=True)  
train_triplets_df = pd.read_csv('train_triplets.txt', header=None, names=['a','b','c'], delim_whitespace=True)  

# Convert to numpy
test_triplets  = test_triplets_df.to_numpy()
train_triplets = train_triplets_df.to_numpy() 


# %% Resize images to a consistent resolution (310,460,3) min=(242,354,3)
# Create the food/food_res directory by hand
"""
print("Start Resizing Images")
tic = time.time()

img_shape_all = np.zeros((10000,3))

for img_nr in range(0,10000):
    image_path        = ''.join(['food/food/',     str(img_nr).zfill(5),'.jpg'])
    image_resize_path = ''.join(['food/food_res/', str(img_nr).zfill(5),'.jpg'])

    img = mpimg.imread(image_path)
    img_shape_all[img_nr,:] = img.shape
    
    mpimg.imsave(image_resize_path, resize(img, (310,460,3),anti_aliasing=True))
    
toc = time.time()
print("Resizing done | Duration = ", toc-tic, "seconds")
"""

# %% Get pretrained model
print("Start Feature Generation")
tic = time.time()

model = VGG16(weights='imagenet', input_shape=(224,224,3), include_top=False, pooling='avg')
nr_of_train_triplets = 50
feat_size = 512 # depends on used model (VGG16: 512)

pos_features = np.zeros((nr_of_train_triplets, feat_size*3))
trp_features = []

i = 1
for trp_nr in range(0,len(train_triplets[0:nr_of_train_triplets,:])):
    for pos in range(0,3):
        img_nr = train_triplets[trp_nr,pos]
        print("Progress: ",str(i),"/",str(nr_of_train_triplets*3))
        i = i + 1;
        
        img_path = ''.join(['food/food/', str(img_nr).zfill(5),'.jpg'])
        img = image.load_img(img_path, target_size=(224,224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        img_features = model.predict(x)
        trp_features = np.append(trp_features, img_features)
        
        
    pos_features[trp_nr,:] = trp_features
    trp_features = []

# After the feature vecotor is generated -> create the inverse one and append
neg_features = np.copy(pos_features)
blk1 = neg_features[:,feat_size:2*feat_size]
blk2 = neg_features[:, 2*feat_size:3*feat_size]

neg_features[:,feat_size:2*feat_size]    = blk2
neg_features[:, 2*feat_size:3*feat_size] = blk1

# Feature Vector
all_features = np.append(pos_features, neg_features ,axis=0)

# Feature Label
all_features_labels = np.ones((nr_of_train_triplets))
all_features_labels = np.append(all_features_labels, np.zeros((nr_of_train_triplets)))


toc = time.time()
print("Feature Generation done | Duration = ", toc-tic, "seconds")

# %% Train classifier for prediction

print("Start StratifiedKFold")
tic = time.time()
n_splits=3
kf = StratifiedKFold(n_splits=n_splits)
X = all_features
y = all_features_labels

clf = RandomForestClassifier(n_estimators=100, random_state=0)
scores_avg = np.zeros((1,4))
clf_keeper = []

i = 0
for train_index, test_index in kf.split(X,y):
    print("Splitting")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    
    # --classification report --
    # print(classification_report(y_test, y_predict, labels=[0,1]))
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_predict, labels=[0,1], average='binary')
    rocs = roc_auc_score(y_test, y_predict)
    
    scores = [precision, recall, f1, rocs]
    
    scores_avg[i,:] = scores_avg[i,:] + np.asarray(scores)/n_splits
    
clf_keeper.append(clf)
i = i+1

scores_df = pd.DataFrame(data=scores_avg,    # values
                         index=['RandomForest'],    # 1st column as index
                         columns=['precision','recall','f1','roc_auc'])  # 1st row as the column names
# Select the classifier with the highest f1 score
clf_best = clf_keeper[np.argmax(scores_df.f1)]

toc = time.time()
print("StratifiedKFold done | Duration = ", toc-tic, "seconds")


# %% Save test labels to csv
# np.savetxt('submission.csv', test_labels, fmt='%1.0f')