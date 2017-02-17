import numpy as np

from utils.Utils import *
from preferentialnet.PNetwork import *
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from dataprovider.cifar import *
from models.MultilayerCNN import *
import pickle

cifarData = CifarData()
cifar = cifarData.read_data_sets()

def standard_scale(X_train, X_test):
    preprocessor=prep.StandardScaler().fit(X_train)
    X_train=preprocessor.transform(X_train)
    X_test=preprocessor.transform(X_test)
    return X_train, X_test


def get_next_block_from_data(data_x, data_y, start_index, batch_size_):
    return data_x[start_index:(start_index + batch_size_)], data_y[start_index:(start_index + batch_size_)]

def get_random_block_from_data(data_x, data_y, batch_size_):
    start_index = np.random.randint(0, len(data_x) - batch_size_)
    return data_x[start_index:(start_index + batch_size_)], data_y[start_index:(start_index + batch_size_)]

X_train, X_test = standard_scale(cifar.train.images, cifar.test.images)
Y_train = cifar.train.labels
Y_test = cifar.test.labels

n_samples = int(cifar.train.num_examples)
n_test_samples = int(cifar.test.num_examples)
training_epochs = 200
eval_epochs = 1
inference_epochs = 100
batch_size = 100
display_step = 1

MODEL_NAME = "cnn"

params = {
    'input_dim': 1024,
    'output_dim': 10,
    'input_image_shape': [-1, 32, 32, 1],
    'network_name': "conv3_local3",
    'batch_size': 100,
    'num_classes': 10,
    'layer_params': [
        {
            'type': 'conv',
            'W_shape': [5, 5, 1, 32],
            'b_shape': [32]
        },
{
            'type': 'conv',
            'W_shape': [5, 5, 32, 64],
            'b_shape': [64]
        },
        # {
        #     'type': 'conv',
        #
        #     'W_shape': [5, 5, 1, 128],
        #     'b_shape': [128]
        # },
        {
            'type': 'local',
            'first': True,
            'W_shape': [8*8*64, 1024],
            'b_shape': [1024]
        },
        {
            'type': 'local',
            'first': False,
            'W_shape': [1024, 512],
            'b_shape': [512]
        },
{
            'type': 'local',
            'first': False,
            'W_shape': [512, 256],
            'b_shape': [256]
        },
{
            'type': 'readout',
            'W_shape': 256
        }
    ],
    'activation_pattern_layers': ['layer_2', 'layer_3', 'layer_4'],
    'num_local_layers': 3,
    'activation_function': tf.nn.relu,
    'optimizer': tf.train.AdamOptimizer(learning_rate=1e-4),
    'tensorboard_dir': 'models/tensorboard_CNN',
}

model = MultilayerCNN(params)

global_step = 1
for epoch in range(training_epochs):
    avg_cost = 0.0
    total_batch = int(n_samples / batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs, batch_ys = get_random_block_from_data(X_train, Y_train, batch_size)

        # Fit training using batch raw_data
        cost, summary = model.partial_fit(batch_xs, batch_ys)
        # Compute average loss
        avg_cost += cost / n_samples * batch_size
        model.summ_writer.add_summary(summary, global_step)
        global_step += 1

    # Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
        #     weights = model.get_weight("W_{0}".format(i))[0]
        #     pr_weights = np.abs(np.sum(weights, 0))
        #     Utils.plot_histogram(pr_weights, 50)


# weights_ = model.get_weight()
# for idx, w in enumerate(weights_):
#     Utils.plot_histogram(data=np.sum(np.abs(w), 0),
#                          title="Weight distribution per nodes",
#                          save_to="results/ff/weight_{1}_{0}.png".format("_".join([str(x) for x in params['network_shape'][1:-1]]), str(idx)))

eval = []
for epoch_ in range(eval_epochs):
    total_batch = int(n_test_samples / batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs, batch_ys = get_next_block_from_data(X_test, Y_test, i*batch_size, batch_size)

        # Fit training using batch raw_data
        accuracy = model.get_accuracy(batch_xs, batch_ys)
        eval.append(accuracy)
        if epoch_ % 100 == 0:
            print("Epoch:", '%04d' % (epoch_ + 1), "accuracy =", "{:.9f}".format(accuracy))

print(eval)
with open("results/{1}/eval_{0}.pickle".format(params['network_name'], MODEL_NAME), 'wb') as act_f:
    pickle.dump(eval, act_f)

activation_dicts = {k: {} for k in params['activation_pattern_layers']}
node_activation_lists = {k: [] for k in params['activation_pattern_layers']}
sum_of_act_dicts = {k: [] for k in params['activation_pattern_layers']}
for epoch in range(inference_epochs):
    total_batch=int(n_test_samples / batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs, batch_ys = get_random_block_from_data(X_test, Y_test, batch_size)

        cost = model.inference(batch_xs, batch_ys)
        activation_patterns = model.get_activations(batch_xs)
        for j, act_patt in activation_patterns.items():
            current_act_dict = activation_dicts[j]
            current_node_activation_list = node_activation_lists[j]
            sum_of_act_dicts[j] += np.mean(act_patt, 1).tolist()
            for single_act_patt in act_patt:
                nonzero_tup = tuple(np.nonzero(single_act_patt)[0])
                binarized_act = np.where(single_act_patt > 0.0, 1.0, 0.0)
                if(len(current_node_activation_list) > 0):
                    current_node_activation_list += binarized_act
                else:
                    current_node_activation_list = binarized_act
                if nonzero_tup in current_act_dict:
                    current_act_dict[nonzero_tup] += 1
                else:
                    current_act_dict[nonzero_tup] = 1
            activation_dicts[j] = current_act_dict
            node_activation_lists[j] = current_node_activation_list

# with open("results/{1}/sum_act_{0}.pickle".format(params['network_name'], MODEL_NAME), 'wb') as sum_act_f:
#     pickle.dump(sum_of_act_dicts, sum_act_f)

with open("results/{1}/node_act_{0}.pickle".format(params['network_name'], MODEL_NAME), 'wb') as node_act_f:
    pickle.dump(node_activation_lists, node_act_f)

# sorted_acts = {}
# for k, act_dict in activation_dicts.items():
#     act_sorted = sorted(set(act_dict.values()), reverse=True)[:1000]
#     sorted_acts[k] = act_sorted
# with open("results/{1}/act_{0}.pickle".format(params['network_name'], MODEL_NAME), 'wb') as act_f:
#     pickle.dump(sorted_acts, act_f)

