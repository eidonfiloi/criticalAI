import tensorflow as tf
from utils.Utils import *


class CriticalRNN(object):
    """
    Class representing simple multilayer feedforward net
    """

    def __init__(self, params, hidden_weights=None):
        self.params = params
        self.network_shape = self.params['network_shape']
        self.input_dim = self.network_shape[0]
        self.output_dim = self.network_shape[-1]
        self.batch_size = self.params['batch_size']
        self.hidden_weights = hidden_weights
        self.weights = self.initialize_weights()
        self.hs = self.initialize_hs()
        self.hidden_states = self.initialize_hidden_states()
        self.tensorboard_dir = self.params['tensorboard_dir']

        self.activation_function = self.params['activation_function']
        self.optimizer_ = self.params['optimizer']
        # model
        self.input = tf.placeholder(tf.float32, [None, self.input_dim])
        self.output = tf.placeholder(tf.float32, [None, self.output_dim])
        self.activation_patterns = {}

        self.activation = self.input
        for i in range(len(self.weights)):
            if i < len(self.weights) - 1:
                act = self.activation_function(tf.add(tf.matmul(self.activation, self.weights['W_{0}'.format(i)]),
                                               tf.matmul(self.hidden_states['hs_{0}'.format(i+1)], self.hs['H_{0}'.format(i+1)])))
                self.hidden_states['hs_{0}'.format(i+1)] = act
            else:
                act = self.activation_function(tf.matmul(self.activation, self.weights['W_{0}'.format(i)]))
            self.activation = act
            self.activation_patterns['activation_layer_{0}'.format(i + 1)] = act

        # cost
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.activation, self.output))
        tf.summary.scalar('cost', self.cost)
        self.optimizer = self.optimizer_.minimize(self.cost)

        self.sess = tf.Session()

        self.merged = tf.summary.merge_all()
        self.summ_writer = tf.train.SummaryWriter(self.tensorboard_dir, self.sess.graph)

        init = tf.initialize_all_variables()

        self.sess.run(init)

    def initialize_hidden_states(self):
        hidden_states = dict()
        for i in range(1, len(self.network_shape) - 1):
            h_state = tf.random_normal([self.batch_size, self.network_shape[i]], stddev=0.01)
            hidden_states["hs_{0}".format(i)] = h_state
            Utils.variable_summaries(h_state)
        return hidden_states

    def initialize_hs(self):
        hidden_weights = dict()
        for i in range(1, len(self.network_shape) - 1):
            if self.hidden_weights is not None and "H_{0}".format(i) in self.hidden_weights.keys():
                h_current = tf.Variable(self.hidden_weights["H_{0}".format(i)].astype('float32'), dtype=tf.float32, trainable=False)
                hidden_weights["H_{0}".format(i)] = h_current
                Utils.variable_summaries(h_current)
            else:
                h_current = tf.Variable(
                    tf.random_normal([self.network_shape[i + 1], self.network_shape[i + 1]], stddev=0.01), trainable=False)
                hidden_weights["H_{0}".format(i)] = h_current
                Utils.variable_summaries(h_current)
        return hidden_weights

    def initialize_weights(self):
        all_weights = dict()
        for i in range(0, len(self.network_shape) - 1):
            w_current = tf.Variable(tf.random_normal([self.network_shape[i], self.network_shape[i + 1]], stddev=0.01))
            all_weights["W_{0}".format(i)] = w_current
            Utils.variable_summaries(w_current)
        return all_weights

    def partial_fit(self, X, Y):
        cost, opt, merged = self.sess.run((self.cost, self.optimizer, self.merged), feed_dict={self.input: X, self.output: Y})
        return cost, merged

    def inference(self, X, Y):
        return self.sess.run(self.cost, feed_dict={self.input: X, self.output: Y})

    def calc_total_cost(self, X, Y):
        return self.sess.run(self.cost, feed_dict={self.input: X, self.output: Y})

    def get_activation_pattern(self, input_, activation_name=None):
        if activation_name is not None:
            return [self.sess.run(self.activation_patterns[activation_name], feed_dict={self.input: input_})]
        else:
            return [self.sess.run(self.activation_patterns['activation_layer_{0}'.format(i + 1)], feed_dict={self.input: input_})
                    for i in range(len(self.weights))]

    def get_weight(self, weight_name=None):
        if weight_name is not None:
            return [self.sess.run(self.weights[weight_name])]
        else:
            return [self.sess.run(self.weights['W_{0}'.format(i)]) for i in range(len(self.weights))]

