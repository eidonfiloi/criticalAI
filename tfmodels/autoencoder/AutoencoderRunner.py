import numpy as np

from utils.Utils import *
from preferentialnet.Network import *
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from tfmodels.autoencoder.autoencoder_models.Autoencoder import Autoencoder

mnist=input_data.read_data_sets('../../data/MNIST_data', one_hot=True)

def standard_scale(X_train, X_test):
    preprocessor=prep.StandardScaler().fit(X_train)
    X_train=preprocessor.transform(X_train)
    X_test=preprocessor.transform(X_test)
    return X_train, X_test


def get_random_block_from_data(data, batch_size):
    start_index=np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]

X_train, X_test=standard_scale(mnist.train.images, mnist.test.images)

n_samples=int(mnist.train.num_examples)
training_epochs=20
batch_size=128
display_step=1

params = {
    'n_input': 784,
    'n_hidden': 500,
    'transfer_function': tf.nn.softplus,
    'optimizer': tf.train.AdamOptimizer(learning_rate=0.001)
}

start_w = np.load("start_w_784.npy")[:, np.random.randint(784, size=500)]
Utils.plot_histogram(np.sum(start_w, 0).tolist(), 50)
autoencoder=Autoencoder(params, start_w)

net_params = {
        'm': 20,
        'maxM': 784
    }

# net = Network(net_params)
# start_w = net.get_weight_matrix()
#
# np.save("start_w_784", start_w)
#
# Utils.plot_histogram(np.sum(start_w, 0).tolist()[0], 50)


for epoch in range(training_epochs):
    avg_cost=0.
    total_batch=int(n_samples / batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs=get_random_block_from_data(X_train, batch_size)

        # Fit training using batch data
        cost=autoencoder.partial_fit(batch_xs)
        # Compute average loss
        avg_cost += cost / n_samples * batch_size

    # Display logs per epoch step
    if epoch % display_step == 0:
        print( "Epoch:", '%04d' % (epoch + 1), \
            "cost=", "{:.9f}".format(avg_cost))
        weights = autoencoder.getWeights()
        pr_weights = np.abs(np.sum(weights, 0))
        rec_weights = autoencoder.getReconstructionWeights()
        pr_rec_weights = np.abs(np.sum(rec_weights, 1))
        total_w = np.add(pr_weights, pr_rec_weights)
        Utils.plot_histogram(total_w, 50)



print ("Total cost: " + str(autoencoder.calc_total_cost(X_test)))

weights = autoencoder.getWeights()
pr_weights = np.abs(np.sum(weights, 0))
rec_weights = autoencoder.getReconstructionWeights()
pr_rec_weights = np.abs(np.sum(rec_weights, 1))
total_w = np.add(pr_weights, pr_rec_weights)
Utils.plot_histogram(total_w, 50)
