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

# Import Keras
# from tensorflow import keras

from skimage.transform import resize
import matplotlib.image as mpimg

# %% Read in data sets and convert to numpy arrays 

# Read the datasets as DataFrames
test_triplets_df  = pd.read_csv('test_triplets.txt',  header=None, names=['a','b','c'], delim_whitespace=True)  
train_triplets_df = pd.read_csv('train_triplets.txt', header=None, names=['a','b','c'], delim_whitespace=True)  

# Convert to numpy
test_triplets  = test_triplets_df.to_numpy()
train_triplets = train_triplets_df.to_numpy() 

# %% Resize images to a consistent resolution (310,460,3) min=(242,354,3)
print("Start Resizing Images")
tic = time.time()

img_shape_all = np.zeros((10000,3))

for img_nr in range(0,10000):
    image_path = ''.join(['food/food/',str(img_nr).zfill(5),'.jpg'])
    image_resize_path = ''.join(['food/food_res/',str(img_nr).zfill(5),'.jpg'])

    img = mpimg.imread(image_path)
    img_shape_all[img_nr,:] = img.shape
    
    mpimg.imsave(image_resize_path, resize(img, (310,460,3),anti_aliasing=True))
    
toc = time.time()
print("Resizing done | Duration = ", toc-tic, "seconds")


# %% Save test labels to csv
# np.savetxt('submission.csv', test_labels, fmt='%1.0f')