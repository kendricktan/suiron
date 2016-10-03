import tensorflow as tf

import tflearn

from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

# NVIDIA's CNN architecture  
# (used for unprocessed data)
def get_cnn_model(checkpoint_path='cnn_model', width=72, height=48, depth=3):
    network = input_data(shape=[None, height, width, depth], name='input')

    # Convolution no.1
    # Relu introduces non linearity into training
    network = conv_2d(network, 24, [5, 3], activation='relu')

    # Convolution no.2
    network = conv_2d(network, 36, [5, 24], activation='relu')
    
    # Convolution no.3
    network = conv_2d(network, 48, [5, 36], activation='relu')

    # Convolution no.4
    network = conv_2d(network, 64, [3, 48], activation='relu')

    # Convolution no.5
    network = conv_2d(network, 64, [3, 64], activation='relu')

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

    network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', 
                        learning_rate=0.001, name='target')

    # Verbosity yay nay
    # 0 = nothing
    model = tflearn.DNN(network, tensorboard_verbose=3, checkpoint_path=checkpoint_path) 
    return model


