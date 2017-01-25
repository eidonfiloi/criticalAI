import tensorflow as tf
from utils.Utils import *


class FeedforwardNet(object):
    """
    Class representing simple multilayer feedforward net
    """

    def __init__(self, params, start_weights=None):
        self.params = params
        self.network_shape = self.params['network_shape']
        self.input_dim = self.network_shape[0]
        self.output_dim = self.network_shape[-1]
        self.start_weights = start_weights
        self.weights = {}
        self.tensorboard_dir = self.params['tensorboard_dir']

        self.activation_function = self.params['activation_function']
        self.optimizer_ = self.params['optimizer']
        # model
        self.input = tf.placeholder(tf.float32, [None, self.input_dim])
        self.output = tf.placeholder(tf.float32, [None, self.output_dim])
        self.activation_patterns = {}

        self.activation = self.input

        for i in range(len(self.network_shape) - 1):
            with tf.name_scope("layer_{0}".format(i+1)):
                W = tf.Variable(tf.truncated_normal(shape=[self.network_shape[i], self.network_shape[i+1]], stddev=0.1),name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.network_shape[i+1]]), name="b")
                self.weights['W_{0}'.format(i+1)] = W
                Utils.variable_summaries(W, "W")
                Utils.variable_summaries(b, "b")
                self.activation = self.activation_function(tf.matmul(self.activation, W) + b)
                self.activation_patterns['activation_layer_{0}'.format(i + 1)] = self.activation

        # for i in range(len(self.weights)):
        #     act = self.activation_function(tf.matmul(self.activation, self.weights['W_{0}'.format(i)]))
        #     self.activation = act
        #     self.activation_patterns['activation_layer_{0}'.format(i + 1)] = act

        # cost

        correct_prediction = tf.equal(tf.argmax(self.activation, 1), tf.argmax(self.output, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.activation, self.output))
        tf.summary.scalar('cost', self.cost)
        self.optimizer = self.optimizer_.minimize(self.cost)

        self.sess = tf.Session()

        self.merged = tf.summary.merge_all()
        self.summ_writer = tf.summary.FileWriter(self.tensorboard_dir, self.sess.graph)

        init = tf.global_variables_initializer()

        self.sess.run(init)

    def initialize_weights(self):
        all_weights = dict()
        for i in range(0, len(self.network_shape) - 1):
            if self.start_weights is not None and "W_{0}".format(i) in self.start_weights.keys():
                w_current = tf.Variable(self.start_weights["W_{0}".format(i)].astype('float32'), dtype=tf.float32)
                all_weights["W_{0}".format(i)] = w_current
                Utils.variable_summaries(w_current)
            else:
                w_current = tf.Variable(tf.random_normal([self.network_shape[i], self.network_shape[i + 1]], stddev=0.01))
                all_weights["W_{0}".format(i)] = w_current
                Utils.variable_summaries(w_current)
        return all_weights

    def partial_fit(self, X, Y):
        cost, opt, merged = self.sess.run((self.cost, self.optimizer, self.merged), feed_dict={self.input: X, self.output: Y})
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
                    for i in range(len(self.weights))]

    def get_weight(self, weight_name=None):
        if weight_name is not None:
            return [self.sess.run(self.weights[weight_name])]
        else:
            return [self.sess.run(v) for _, v in self.weights.items()]

