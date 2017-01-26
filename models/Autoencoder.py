import tensorflow as tf
import numpy as np
from tfmodels.autoencoder.Utils import *
from utils.Utils import *


class Autoencoder(object):

    def __init__(self, params):
        self.params = params
        self.network_shape = params['network_shape']
        self.tensorboard_dir = params['tensorboard_dir']
        self.input_dim = self.network_shape[0]
        self.output_dim = self.network_shape[-1]
        self.activation_function = self.params['activation_function']
        self.optimizer_ = self.params['optimizer']
        self.weights = {}

        self.input = tf.placeholder(tf.float32, [None, self.input_dim])
        self.output = tf.placeholder(tf.float32, [None, self.output_dim])

        self.activation_patterns = {}
        self.layer_output = self.input

        with tf.name_scope("layer_1"):
            self.layer_input = self.layer_output
            with tf.name_scope("hidden"):
                W_in = tf.Variable(tf.truncated_normal(shape=[self.network_shape[0], self.network_shape[1]], stddev=0.1), name="W_in")
                Utils.variable_summaries(W_in, "W_in")
                self.weights["W_in"] = W_in
                b_in = tf.Variable(tf.constant(shape=[self.network_shape[1]], value=0.1), name="b_in")
                Utils.variable_summaries(W_in, "b_in")
                self.hidden = self.activation_function(tf.matmul(self.layer_output, W_in) + b_in)
                Utils.variable_summaries(self.hidden, "hidden")
                self.activation_patterns["layer_{0}".format(1)] = self.hidden
            with tf.name_scope("reconstruction"):
                W_out = tf.Variable(tf.truncated_normal(shape=[self.network_shape[1], self.network_shape[0]], stddev=0.1), name="W_out")
                Utils.variable_summaries(W_in, "W_out")
                self.weights["W_out"] = W_out
                b_out = tf.Variable(tf.constant(shape=[self.network_shape[0]], value=0.1), name="b_out")
                Utils.variable_summaries(W_in, "b_out")
                self.reconstruction = self.activation_function(tf.matmul(self.hidden, W_out) + b_out)
            self.layer_output = self.hidden

        self.cost = tf.reduce_sum(tf.pow(tf.sub(self.reconstruction, self.layer_input), 2.0),
                                  name="reconstruction_cost")
        tf.summary.scalar('cost', self.cost)
        self.optimizer = self.optimizer_.minimize(self.cost)

        self.sess = tf.Session()

        self.merged = tf.summary.merge_all()
        self.summ_writer = tf.summary.FileWriter(self.tensorboard_dir, self.sess.graph)

        init = tf.global_variables_initializer()

        self.sess.run(init)

    def partial_fit(self, X):
        cost, opt, merged = self.sess.run((self.cost, self.optimizer, self.merged), feed_dict={self.input: X})
        return cost, merged

    def inference(self, X):
        return self.sess.run(self.cost, feed_dict={self.input: X})

    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict = {self.input: X})

    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict={self.input: X})

    def generate(self, hidden = None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights["b1"])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.input: X})

    def get_hidden_activation(self, X):

        return [self.sess.run(self.hidden, feed_dict={self.input: X})]

    def get_activations(self, X):
        return {k: self.sess.run(v, feed_dict={self.input: X}) for k, v in self.activation_patterns.items()}

    def get_weights(self):
        return {k: self.sess.run(v) for k, v in self.weights.items()}

    def get_accuracy(self, X):
        return self.sess.run(self.cost, feed_dict={self.input: X})