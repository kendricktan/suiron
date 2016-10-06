import tensorflow as tf

import tflearn

from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

# NVIDIA's CNN architecture  
# (used for unprocessed data)
def get_cnn_model(checkpoint_path='cnn_servo_model', width=72, height=48, depth=3, session=None):
    
    # Inputs
    network = input_data(shape=[None, height, width, depth], name='input')

    # Convolution no.1
    # Relu introduces non linearity into training
    network = conv_2d(network, 8, [5, 3], activation='relu')

    # Convolution no.2
    network = conv_2d(network, 12, [5, 8], activation='relu')
    
    # Convolution no.3
    network = conv_2d(network, 16, [5, 16], activation='relu')

    # Convolution no.4
    network = conv_2d(network, 24, [3, 20], activation='relu')

    # Convolution no.5
    network = conv_2d(network, 24, [3, 24], activation='relu')

    # Fully connected no.1
    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    # Fully connected no.2
    network = fully_connected(network, 100, activation='relu')
    network = dropout(network, 0.8)

    # Fully connected no.3
    network = fully_connected(network, 50, activation='relu')
    network = dropout(network, 0.8)

    # Fully connected no.4
    network = fully_connected(network, 10, activation='relu')
    network = dropout(network, 0.8)
 
    # Fully connected no.5
    network = fully_connected(network, 1, activation='tanh')

    # Regression
    network = regression(network, loss='mean_square', metric='accuracy', learning_rate=1e-4,name='target') 

    # Verbosity yay nay
    # 0 = nothing
    model = tflearn.DNN(network, tensorboard_verbose=2, checkpoint_path=checkpoint_path, session=session) 
    return model

# Simple One hidden layer NN to determine the (linear?) relationship between 
# the steering angle and motor value
def get_nn_model(checkpoint_path='nn_motor_model', session=None):
    # Input is a single value (raw motor value)
    network = input_data(shape=[None, 1], name='input')

    # Hidden layer no.1,  
    network = fully_connected(network, 12, activation='linear')
    
    # Output layer
    network = fully_connected(network, 1, activation='tanh')

    # regression
    network = regression(network, loss='mean_square', metric='accuracy', name='target')

    # Verbosity yay nay
    model = tflearn.DNN(network, tensorboard_verbose=3, checkpoint_path=checkpoint_path, session=session)
    return model