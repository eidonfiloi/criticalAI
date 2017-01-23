import numpy as np

from utils.Utils import *
from preferentialnet.PNetwork import *
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from models.CriticalRNN import *
import pickle

from tfmodels.autoencoder.autoencoder_models.Autoencoder import Autoencoder

mnist=input_data.read_data_sets('raw_data/MNIST_data', one_hot=True)


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

n_samples = int(mnist.train.num_examples)
training_epochs = 150
inference_epochs = 20
batch_size = 128
display_step = 1

params = {
    'network_shape': [784, 400, 200, 10],
    'batch_size': batch_size,
    'activation_function': tf.nn.relu,
    'optimizer': tf.train.AdamOptimizer(learning_rate=1e-4),
    'tensorboard_dir': 'models/tensorboard_criticalRNN'
}

hidden_weight_dict = {}
for i in range(1, len(params['network_shape']) - 1):
    start_w = np.load("raw_data/hidden_h_{0}.npy".format(params['network_shape'][i])).astype('float32')
    # start_w -= 0.0000001
    hidden_weight_dict['H_{0}'.format(i)] = start_w
    Utils.plot_histogram(np.sum(start_w, 0).tolist(), 50)

model = CriticalRNN(params, hidden_weight_dict)

activation_dicts = [{} for i in range(1, len(params['network_shape']))]
for epoch in range(training_epochs):
    avg_cost=0.0
    total_batch=int(n_samples / batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs, batch_ys = get_random_block_from_data(X_train, Y_train, batch_size)

        # Fit training using batch raw_data
        cost, summary = model.partial_fit(batch_xs, batch_ys)
        # Compute average loss
        avg_cost += cost / n_samples * batch_size
        model.summ_writer.add_summary(summary, i)

    # Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
        # weights = model.get_weight("W_{0}".format(i))[0]
        # pr_weights = np.abs(np.sum(weights, 0))
        # Utils.plot_histogram(pr_weights, 50)

# print("Total cost: " + str(model.calc_total_cost(X_test, Y_test)))

for epoch_ in range(inference_epochs):
    total_batch=int(n_samples / batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs, batch_ys = get_random_block_from_data(X_train, Y_train, batch_size)

        # Fit training using batch raw_data
        cost = model.inference(batch_xs, batch_ys)
        if epoch_ % 100 == 0:
            print("Epoch:", '%04d' % (epoch_ + 1), "cost=", "{:.9f}".format(cost))
        activation_patterns = model.get_activation_pattern(batch_xs)
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
with open("raw_data/criticalRNN_states_top1000_400_200_100.pickle", 'wb') as act_f:
    pickle.dump(sorted_acts, act_f)
