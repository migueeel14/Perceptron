# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Load libraries
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
from NN_functions import *

#%%
# setting the random seed to reproduce results
np.random.seed(5)

# number of observations
obs = 300

# generating synthetic data from multivariate normal distribution  
class_zeros = np.random.multivariate_normal([0,0], [[1.,.95],[.95,1.]], obs)
class_ones = np.random.multivariate_normal([1,5], [[1.,.85],[.85,1.]], obs)

# generating a column of ones as a dummy feature to create an intercept
intercept = np.ones((2*obs,1))

# vertically stacking the two classes 
features = np.vstack((class_zeros, class_ones)).astype(np.float32)

# putting in the dummy feature column
#features = np.hstack((intercept, features))

# creating the labels for the two classes
label_zeros = np.zeros((obs,1))
label_ones = np.ones((obs,1))

# stacking the labels, and then adding them to the dataset
labels = np.vstack((label_zeros,label_ones))
#plt.scatter(features[:,0], features[:,1], c = labels[:,0])
dataset = np.hstack((features,labels))

# scatter plot to visualize the two classes (red=1, blue=0)


X = features.T
Y = labels.T


#%%
# shuffling the data to make the sampling random
np.random.shuffle(dataset)

# splitting the data into train/test sets
train = dataset[0:int(0.7*(obs*2))]
test = dataset[int(0.7*(obs*2)):(obs*2)]

print('Size data = ',len(dataset), '\nSize training data = ',len(train),'\nSize test data = ',len(test))
X_train = train[:,0:2]
X_test  = test[:,0:2]
Y_train = train[:,2:3]
Y_test  = test[:,2:3]

shape_X = X_train.shape
shape_Y = Y_train.shape
m = X_train.shape[1]

print ('The shape of X is: ' + str(shape_X))
print ('The shape of Y is: ' + str(shape_Y))
print ('I have m = %d training examples!' % (m))


#%%
# Define NN to use ('adaline','log_reg', 'relu')
NN = 'relu' 

train_set_x = X_train.T
test_set_x = X_test.T
train_set_y = Y_train.T
test_set_y = Y_test.T
#%%
learning_rates = [0.05, 0.01, 0.5, 0.1, 1]
models = {}
for i in learning_rates:
    print ("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, NN, num_iterations = 5000, learning_rate = i, print_cost = False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations (hundreds)')
plt.title(NN)
legend = plt.legend(loc='upper center', shadow=True)
# Comentario prueba, borrar luego
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()
