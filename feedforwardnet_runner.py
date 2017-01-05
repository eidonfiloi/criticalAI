import numpy as np

from utils.Utils import *
from preferentialnet.Network import *
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from models.FeedforwardNet import *

from tfmodels.autoencoder.autoencoder_models.Autoencoder import Autoencoder

mnist=input_data.read_data_sets('data/MNIST_data', one_hot=True)


def standard_scale(X_train, X_test):
    preprocessor=prep.StandardScaler().fit(X_train)
    X_train=preprocessor.transform(X_train)
    X_test=preprocessor.transform(X_test)
    return X_train, X_test


def get_random_block_from_data(data_x, data_y, batch_size_):
    start_index = np.random.randint(0, len(data_x) - batch_size_)
    return data_x[start_index:(start_index + batch_size_)], data_y[start_index:(start_index + batch_size_)]

X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)
Y_train = mnist.train.labels
Y_test = mnist.test.labels

n_samples=int(mnist.train.num_examples)
training_epochs=20
batch_size=128
display_step=1

params = {
    'network_shape': [784, 400, 200, 10],
    'activation_function': tf.nn.relu,
    'optimizer': tf.train.AdamOptimizer(learning_rate=0.001)
}

# start_w = np.load("start_w_784.npy")[:, np.random.randint(784, size=500)]
# Utils.plot_histogram(np.sum(start_w, 0).tolist(), 50)
# autoencoder=Autoencoder(params, start_w)

model = FeedforwardNet(params)

for epoch in range(training_epochs):
    avg_cost=0.0
    total_batch=int(n_samples / batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs, batch_ys = get_random_block_from_data(X_train, Y_train, batch_size)

        # Fit training using batch data
        cost = model.partial_fit(batch_xs, batch_ys)
        # Compute average loss
        avg_cost += cost / n_samples * batch_size

    # Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
        weights = model.get_weight("W_1")
        pr_weights = np.abs(np.sum(weights, 0))
        total_w = np.add(pr_weights, pr_weights)
        Utils.plot_histogram(total_w, 50)

print("Total cost: " + str(model.calc_total_cost(X_test, Y_test)))

weights = model.get_weight(1)
pr_weights = np.abs(np.sum(weights, 0))
Utils.plot_histogram(pr_weights, 50)
