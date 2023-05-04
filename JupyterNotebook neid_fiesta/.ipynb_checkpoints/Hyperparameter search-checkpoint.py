import sys
sys.path.append('../src')

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import numpy as np
import matplotlib.pyplot as plt
from PyAstronomy import pyasl
from datetime import datetime

from FIESTA_functions import *
from HARPS_N_functions import *
from NEID_solar_functions import *
from functions import *

from multiprocessing import Pool
import time
import math
from scipy import optimize
import scipy.optimize as opt


import tensorflow as tf
import keras
from keras import layers
from keras import models
from keras import utils
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Activation
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras import datasets

from keras.callbacks import LearningRateScheduler
from keras.callbacks import History

from keras import losses
from sklearn.utils import shuffle

# print(tf.VERSION)
print(tf.keras.__version__)

# Use scikit-learn to grid search the batch size and epochs
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.losses import mean_squared_error
# Function to create model, required for KerasClassifier


TF_ENABLE_ONEDNN_OPTS=0
#----------------------------------
# Read data
#----------------------------------
RV_FT_k     = np.loadtxt('../lib/FIESTA_daily_output/RV_FT_k.txt')
eRV_FT_k    = np.loadtxt('../lib/FIESTA_daily_output/eRV_FT_k.txt')
ΔRV_k       = np.loadtxt('../lib/FIESTA_daily_output/ΔRV_k.txt')
bjd_daily   = np.loadtxt('../lib/FIESTA_daily_output/bjd_daily.txt')
rv_daily    = np.loadtxt('../lib/FIESTA_daily_output/rv_daily.txt')
σrv_daily   = np.loadtxt('../lib/FIESTA_daily_output/σrv_daily.txt')
A_k         = np.loadtxt('../lib/FIESTA_daily_output/A_k.txt')

RV_inj = np.random.rand(len(rv_daily))*2
X = (RV_FT_k[:5] + RV_inj).T
Y = RV_inj


import numpy as np
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor

def create_model():

	def custom_loss(y_test, y_pred):
		squared_difference = tf.dtypes.cast(tf.square((y_test - y_pred)*weights), tf.float64) 
		return tf.dtypes.cast(tf.math.reduce_sum(squared_difference)/tf.math.reduce_sum(tf.square(weights)), tf.float64)  # Note the `axis=-1`

	# def custom_loss(y_true, y_pred):
	# 	return np.mean(weights * np.abs(y_true - y_pred))
    
	# create model
	model = Sequential()
	model.add(Dense(1, input_shape=(5,), activation=tf.keras.activations.linear))
    
    # Compile model
	model.compile(loss=custom_loss,
                  optimizer='adam',
                  # metrics=['mean_squared_error']
                 )
	return model
# fix random seed for reproducibility
seed = 7
tf.random.set_seed(seed)

# create model
model = KerasRegressor(model=create_model, epochs=1000, batch_size=202, verbose=0)
# define the grid search parameters
# optimizer = ['adam']

if 1:
    '''
        Best: 0.910569 using {'batch_size': 100, 'epochs': 2000, 'optimizer__learning_rate': 0.3, 'optimizer__momentum': 0.9}
    '''
    batch_size = [10, 50, 100, 202] 
    epochs = [500, 1000, 2000]
    learn_rate = [0.0001, 0.001, 0.01, 0.1, 0.3]
    momentum = [0.0, 0.3, 0.6, 0.9]
    param_grid = dict(batch_size=batch_size, epochs=epochs, optimizer__learning_rate=learn_rate, optimizer__momentum=momentum)

if 0:
    batch_size = [10, 50, 100, 202] 
    param_grid = dict(batch_size=batch_size, epochs=[2000], optimizer__learning_rate=[0.001], optimizer__momentum=[0.9])

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
weights = 1/σrv_daily**2
# grid_result = grid.fit(X, Y, sample_weight=weights)
grid_result = grid.fit(X, Y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))