import tensorflow as tf
from utils.Utils import *


class CriticalEchoNet(object):
    """
    Class representing a critical echo net
    """

    def __init__(self, params, hidden_weights=None):
        self.params = params
        self.network_shape = self.params['network_shape']
        self.input_dim = self.network_shape[0]
        self.output_dim = self.network_shape[-1]
        self.batch_size = self.params['batch_size']
        self.hidden_weights = hidden_weights
        # self.weights = self.initialize_weights()
        # self.mirror_weights = self.initialize_mirror_weights()
        # self.readout_weights = self.initialize_readout_weights()
        # self.hs = self.initialize_hs()
        # self.hidden_states = self.initialize_hidden_states()
        self.tensorboard_dir = self.params['tensorboard_dir']

        self.activation_function = self.params['activation_function']
        self.optimizer_ = self.params['optimizer']
        # model
        self.input = tf.placeholder(tf.float32, [None, self.input_dim], name="input")
        self.output = tf.placeholder(tf.float32, [None, self.output_dim], name="output")
        self.activation_patterns = {}
        self.hidden_state_activation_patterns = {}
        self.activation = self.input
        self.hidden_states = {}
        self.hidden_states_update_ops = {}
        for i in range(1, len(self.network_shape) - 1):
            h = tf.Variable(tf.truncated_normal([self.batch_size, self.network_shape[i]]), name="hs_{0}".format(i), trainable=False)
            self.hidden_states["hs_{0}".format(i)] = h
            Utils.variable_summaries(self.hidden_states["hs_{0}".format(i)], "hs_{0}".format(i))
        if self.hidden_weights is not None:
            H_tune = tf.Variable(1.0, trainable=True, name="H_tune")
            Utils.variable_summaries(H_tune, "H_tune")
        else:
            H_tune = tf.Variable(1, trainable=False, name="H_tune")
        for i in range(len(self.network_shape) - 1):
            with tf.name_scope("layer{0}".format(i + 1)):
                # input weight and bias
                W = tf.Variable(tf.random_normal([self.network_shape[i], self.network_shape[i + 1]], stddev=0.01),
                                name="W")
                bW = tf.Variable(tf.random_normal([self.network_shape[i + 1]], stddev=0.01), name="bW")
                Utils.variable_summaries(W, "W")
                Utils.variable_summaries(bW, "bW")
                if i < len(self.network_shape) - 2:
                    # hidden state power lawed hidden connections
                    H_name = "H_{0}".format(i+1)
                    if self.hidden_weights is not None and H_name in self.hidden_weights.keys():
                        H = tf.Variable(self.hidden_weights[H_name].astype('float32'), dtype=tf.float32, trainable=False, name="H")

                    else:
                        H = tf.Variable(tf.random_normal([self.network_shape[i + 1], self.network_shape[i + 1]], stddev=0.01), trainable=False, name="H")

                    # readout weights and biases
                    R = tf.Variable(tf.random_normal([self.network_shape[i + 1], self.network_shape[i + 1]], stddev=0.01), name="R")
                    bR = tf.Variable(tf.random_normal([self.network_shape[i + 1]], stddev=0.01), name="bR")
                    Utils.variable_summaries(R, "R")
                    Utils.variable_summaries(bR, "bR")

                    with tf.name_scope("hidden"):
                        input_for_hidden = tf.matmul(self.activation, W) + bW
                        hidden_update = self.activation_function(tf.add(input_for_hidden, tf.matmul(self.hidden_states["hs_{0}".format(i+1)], tf.scalar_mul(H_tune, H))))
                        self.hidden_states_update_ops["hs_{0}".format(i + 1)] = self.hidden_states["hs_{0}".format(i + 1)].assign(hidden_update)
                    with tf.name_scope("readout"):
                        self.activation = self.activation_function(tf.matmul(hidden_update, R) + bR)
                    # self.hidden_state_activation_patterns['hidden_state_layer_{0}'.format(i + 1)] = self.hidden_states[self.hidden_states["hs_{0}".format(i+1)]]

                else:
                    with tf.name_scope("activation"):
                        self.activation = self.activation_function(tf.matmul(self.activation, W) + bW)
                act = self.activation
                self.activation_patterns['activation_layer_{0}'.format(i + 1)] = act

        # cost
        with tf.name_scope("cost"):
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.activation, self.output))
            tf.summary.scalar('cost', self.cost)

        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(tf.argmax(self.activation, 1), tf.argmax(self.output, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.optimizer = self.optimizer_.minimize(self.cost)

        self.sess = tf.Session()

        self.merged = tf.summary.merge_all()
        self.summ_writer = tf.summary.FileWriter(self.tensorboard_dir, self.sess.graph)

        init = tf.global_variables_initializer()

        self.sess.run(init)

    def partial_fit(self, X, Y):
        fetches = [self.cost, self.optimizer, self.merged]
        for _, v in self.hidden_states_update_ops.items():
            fetches.append(v)
        cost, opt, merged, *_ = self.sess.run(fetches, feed_dict={self.input: X, self.output: Y})

        return cost, merged

    def inference(self, X, Y):
        return self.sess.run(self.cost, feed_dict={self.input: X, self.output: Y})

    def get_accuracy(self, X, Y):
        return self.sess.run(self.accuracy, feed_dict={self.input: X, self.output: Y})

    def calc_total_cost(self, X, Y):
        return self.sess.run(self.cost, feed_dict={self.input: X, self.output: Y})

    def get_activation_pattern(self, input_, activation_name=None):
        if activation_name is not None:
            return [self.sess.run(self.activation_patterns[activation_name], feed_dict={self.input: input_})]
        else:
            return [self.sess.run(self.activation_patterns['activation_layer_{0}'.format(i + 1)], feed_dict={self.input: input_})
                    for i in range(len(self.network_shape) - 1)]

    def get_hidden_state_pattern(self, input_, activation_name=None):
        if activation_name is not None:
            return [self.sess.run(self.activation_patterns[activation_name], feed_dict={self.input: input_})]
        else:
            return [self.sess.run(self.hidden_state_activation_patterns['hidden_state_layer_{0}'.format(i + 1)], feed_dict={self.input: input_})
                    for i in range(len(self.network_shape) - 1)]

    def get_weight(self, weight_name=None):
        if weight_name is not None:
            return [self.sess.run(self.weights[weight_name])]
        else:
            return [self.sess.run(self.weights['W_{0}'.format(i)]) for i in range(len(self.weights))]

