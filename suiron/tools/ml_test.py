from suiron.core.SuironML import get_cnn_model, get_nn_model

# Test CNN with MNIST dataset
def test_with_mnist():
    import tflearn.datasets.mnist as mnist

    X, Y, testX, testY = mnist.load_data(one_hot=True)
    X = X.reshape([-1, 28, 28, 1])
    testX = testX.reshape([-1, 28, 28, 1])

    model = get_cnn_model(width=28, height=28, depth=1)
    model.fit({'input': X}, {'target': Y}, n_epoch=20,
                validation_set=({'input': testX}, {'target': testY}),
                snapshot_step=100, show_metric=True, run_id='cnn_mnist')

test_with_mnist()
