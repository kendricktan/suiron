import tflearn

from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

# Gets a 2 layered CNN
# (used for unprocessed data)
def get_cnn_model(checkpoint_path='cnn_model', width=72, height=48, depth=3, outputs=10):
    network = input_data(shape=[None, height, width, depth], name='input')

    # Convolution no.1
    # Max pooling does feature extraction
    # Relu introduces non linearity into training
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    # Convolution no.2
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    
    # Convolution no.3
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    # Dropout generalizes better
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, outputs, activation='softmax')

    network = fully_connected(network, outputs, activation='softmax')
    network = regression(network, optimizer='momentum', 
                        loss='categorical_crossentropy', 
                        learning_rate=0.001, name='target')

    # Verbosity yay nay
    # 0 = nothing
    model = tflearn.DNN(network, tensorboard_verbose=3, checkpoint_path=checkpoint_path) 
    return model


# Gets a one layered NN
# (Used for preprocessed data)
def get_nn_model(width=72, height=48, depth=1, outputs=10):
   network = input_data(shape=[None, height, width, depth])
   network = fully_connected(network, height*2, activation='linear') 
   network = dropout(network, 0.5)
   network = fully_connected(network, outputs, activation='linear')
   network = regression(network, optimizer='sdg', loss='mean_square', learning_rate=0.1)

   model = tflearn.DNN(network, tensorboard_verbose=3, checkpoint_path='nn_model')
   return model 