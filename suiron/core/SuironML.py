import tflearn

from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

# Gets a 2 layered CNN
# (used for unprocessed data)
def get_cnn_model(width=72, height=48, depth=3, outputs=10):
    network = input_data(shape=[None, height, width, depth]])

    # Convolution layer no.1
    # ReLU was used as it introduces non-linearity
    network = conv_2d(network, height, 3, activation='relu')
    network = max_pool_2d(network, 2) # Dimensionality reduction and feature extraction

    # Convolution layer no.2
    network = conv_2d(network, height*2, 3, activation='relu')
    network = conv_2d(network, height*2, 3, activation='relu')
    network = max_pool_2d(network, 2)

    # Feed forward/fully connected layer
    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.5) # Reduces overfitting

    network = fully_connected(network, outputs, activation='softmax')
    network = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)

    # Verbosity yay nay
    # 0 = nothing
    model = tflearn.DNN(network, tensorboard_verbose=3) 
    return model


# Gets a one layered NN
# (Used for preprocessed data)
def get_nn(width=72, height=48, depth=1, outputs=10):
   network = input_data(shape=[None, height, width, depth])
   network = fully_connected(network, height*2, activation='linear') 
   network = dropout(network, 0.5)
   network = fully_connected(network, outputs, activation='linear')
   network = regression(network, optimizer='sdg', loss='mean_square', learning_rate=0.01)

   model = tflearn.DNN(network, tensorboard_verbose=3)
   return model 