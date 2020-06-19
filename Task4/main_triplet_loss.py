# -*- coding: utf-8 -*-
"""
Introduction to Machine Learning: Task4

Marco Dober & Vukasin Lalic aka Snorlax

Triplet loss approach 

Idea:
-- Fine-tune pr-etrained model with triplet loss
-- Extract features from fine-tuned model 
-- Put features in binary classifier 

"""
# %% Import libraries 
# General stuff 
import tensorflow as tf
from tensorflow import keras
import numpy as np 
import pandas as pd
import time

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

# %% triplet loss function 
def loss_fn(triplet_embedded, margin):
    anchor_output = triplet_embedded[0,:]
    positive_output = triplet_embedded[1,:]
    negative_output = triplet_embedded[2,:]

    d_pos = tf.reduce_sum(tf.square(anchor_output - positive_output), -1)
    d_neg = tf.reduce_sum(tf.square(anchor_output - negative_output), -1)
    
    loss = tf.maximum(np.float64(0), margin + d_pos - d_neg)
    loss = tf.reduce_mean(loss)
     
    return loss

# %% Read in data sets and convert to numpy arrays 

# Read the datasets as DataFrames
test_triplets_df  = pd.read_csv('test_triplets.txt',  header=None, names=['a','b','c'], delim_whitespace=True)  
train_triplets_df = pd.read_csv('train_triplets.txt', header=None, names=['a','b','c'], delim_whitespace=True)  

# Convert to numpy
test_triplets  = test_triplets_df.to_numpy()
train_triplets = train_triplets_df.to_numpy() 

# %% Load base model 
model = VGG16(weights='imagenet',
              input_shape=(224,224,3),
              include_top=False,
              pooling='avg')

# %% fine tune model 
feat_size = 512 # depends on used model (VGG16: 512)
margin = 0.1
optimizer = keras.optimizers.Adam(1e-5)

for trp_nr in range(0,10):
    print("\rProgress: ",str(trp_nr),"/",str(10), end='\r', flush=True)
    triplet_embedded = np.zeros((3,feat_size))
          
    for abc in range(0,3):
        img_nr = train_triplets[trp_nr, abc]
        img_path = ''.join(['food/', str(img_nr).zfill(5),'.jpg'])
        img = image.load_img(img_path, target_size=(224,224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        with tf.GradientTape() as tape:
            triplet_embedded[abc,:] = model.predict(x)

    loss = loss_fn(triplet_embedded, margin)
    
    # Get gradients of loss wrt the *trainable* weights.
    gradients = tape.gradient(loss, model.trainable_weights)
    # Update the weights of the model.
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    

        
        
        
        
        