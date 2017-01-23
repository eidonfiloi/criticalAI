import numpy as np

from utils.Utils import *
from preferentialnet.PNetwork import *
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pickle

from tfmodels.autoencoder.autoencoder_models.Autoencoder import Autoencoder

mnist = input_data.read_data_sets('../../raw_data/MNIST_data', one_hot=True)


def standard_scale(X_train, X_test):
    preprocessor=prep.StandardScaler().fit(X_train)
    X_train=preprocessor.transform(X_train)
    X_test=preprocessor.transform(X_test)
    return X_train, X_test


def get_random_block_from_data(data, batch_size):
    start_index=np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]

X_train, X_test=standard_scale(mnist.train.images, mnist.test.images)

n_samples = int(mnist.train.num_examples)
training_epochs = 200
inference_epochs = 20
batch_size = 128
display_step = 1

params = {
    'n_input': 784,
    'n_hidden': 400,
    'transfer_function': tf.nn.relu,
    'optimizer': tf.train.AdamOptimizer(learning_rate=1e-4)
}

# start_w = np.load("../../raw_data/start_w_784.npy")[:, np.random.randint(784, size=500)]
# Utils.plot_histogram(np.sum(start_w, 0).tolist(), 50)
# autoencoder = Autoencoder(params, start_w)
autoencoder = Autoencoder(params)

# net_params = {
#         'm': 20,
#         'maxM': 784
#     }
#
# net = Network(net_params)
# start_w = net.get_weight_matrix()
#
# np.save("start_w_784", start_w)
#
# Utils.plot_histogram(np.sum(start_w, 0).tolist()[0], 50)

activation_dicts = [{}]
for epoch in range(training_epochs):
    avg_cost=0.
    total_batch=int(n_samples / batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs=get_random_block_from_data(X_train, batch_size)

        # Fit training using batch raw_data
        cost = autoencoder.partial_fit(batch_xs)
        # Compute average loss
        avg_cost += cost / n_samples * batch_size
        # activation_patterns = autoencoder.get_hidden_activation(batch_xs)
        # for j, act_patt in enumerate(activation_patterns):
        #     current_act_dict = activation_dicts[j]
        #     for single_act_patt in act_patt:
        #         nonzero_tup = tuple(np.nonzero(single_act_patt)[0])
        #         if nonzero_tup in current_act_dict:
        #             current_act_dict[nonzero_tup] += 1
        #         else:
        #             current_act_dict[nonzero_tup] = 1
        #     activation_dicts[j] = current_act_dict
        # print(activation_dicts)

    # Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
        # weights = autoencoder.getWeights()
        # pr_weights = np.abs(np.sum(weights, 0))
        # rec_weights = autoencoder.getReconstructionWeights()
        # pr_rec_weights = np.abs(np.sum(rec_weights, 1))
        # total_w = np.add(pr_weights, pr_rec_weights)
        # Utils.plot_histogram(total_w, 50)

print("Total cost: " + str(autoencoder.calc_total_cost(X_test)))

for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(n_samples / batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)

        # Fit training using batch raw_data
        cost = autoencoder.inference(batch_xs)
        # Compute average loss
        avg_cost += cost / n_samples * batch_size
        if i % 100 == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
        activation_patterns = autoencoder.get_hidden_activation(batch_xs)
        for j, act_patt in enumerate(activation_patterns):
            current_act_dict = activation_dicts[j]
            for single_act_patt in act_patt:
                nonzero_tup = tuple(np.nonzero(single_act_patt)[0])
                if nonzero_tup in current_act_dict:
                    current_act_dict[nonzero_tup] += 1
                else:
                    current_act_dict[nonzero_tup] = 1
            activation_dicts[j] = current_act_dict

sorted_acts = []
for act_dict in activation_dicts:
    act_sorted = sorted(set(act_dict.values()), reverse=True)[:1000]
    sorted_acts.append(act_sorted)
with open("../../raw_data/autoencoder_hidden_top1000_400.pickle", 'wb') as act_f:
    pickle.dump(sorted_acts, act_f)

# weights = autoencoder.getWeights()
# pr_weights = np.abs(np.sum(weights, 0))
# rec_weights = autoencoder.getReconstructionWeights()
# pr_rec_weights = np.abs(np.sum(rec_weights, 1))
# total_w = np.add(pr_weights, pr_rec_weights)
# Utils.plot_histogram(total_w, 50)
