import numpy as np
import tensorflow as tf
from utils.Utils import *


class MultilayerCNN(object):
    """

    """

    def __init__(self, params):
        self.params = params

        self.input_dim = self.params['input_dim']
        self.output_dim = self.params['output_dim']

        self.tensorboard_dir = self.params['tensorboard_dir']

        self.activation_function = self.params['activation_function']
        self.optimizer_ = self.params['optimizer']
        # model
        self.input = tf.placeholder(tf.float32, [None, self.input_dim])
        self.output = tf.placeholder(tf.float32, [None, self.output_dim])

        x_image = tf.reshape(self.input, [-1, 28, 28, 1])

        W_conv1 = Utils.weight_variable([5, 5, 1, 32])
        b_conv1 = Utils.bias_variable([32])

        h_conv1 = self.activation_function(Utils.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = Utils.max_pool_2x2(h_conv1)

        W_conv2 = Utils.weight_variable([5, 5, 32, 64])
        b_conv2 = Utils.bias_variable([64])

        h_conv2 = self.activation_function(Utils.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = Utils.max_pool_2x2(h_conv2)

        W_fc1 = Utils.weight_variable([7 * 7 * 64, 1024])
        b_fc1 = Utils.bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = self.activation_function(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        W_fc2 = Utils.weight_variable([1024, 10])
        b_fc2 = Utils.bias_variable([10])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        self.h_conv1_flat = tf.reshape(h_conv1, [-1, 14 * 14 * 32])
        self.h_conv2_flat = tf.reshape(h_conv2, [-1, 7 * 7 * 64])
        self.h_fc1 = h_fc1
        self.h_fc2 = y_conv

        self.activation_dict = {'h_conv1': self.h_conv1_flat,
                                'h_conv2': self.h_conv2_flat,
                                'h_fc1': self.h_fc1,
                                'h_fc2': self.h_fc2}

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, self.output))
        tf.summary.scalar('cross_entropy', self.cross_entropy)
        self.train_step = self.optimizer_.minimize(self.cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(self.output, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', self.accuracy)

        self.sess = tf.Session()
        self.merged = tf.summary.merge_all()
        self.summ_writer = tf.train.SummaryWriter(self.tensorboard_dir, self.sess.graph)
        init = tf.initialize_all_variables()
        self.sess.run(init)

    def partial_fit(self, X, Y):
        cost, opt, merged = self.sess.run((self.cross_entropy, self.train_step, self.merged),
                                  feed_dict={self.input: X, self.output: Y, self.keep_prob: 0.5})
        return cost, merged

    def get_activations(self, X):
        return {k: self.sess.run(v, feed_dict={self.input: X, self.keep_prob: 1.0}) for k, v in self.activation_dict.items()}

    def inference(self, X, Y):
        accuracy = self.sess.run(self.accuracy, feed_dict={self.input: X, self.output: Y, self.keep_prob: 1.0})
        return accuracy

    def get_accuracy(self, X, Y):
        return self.sess.run(self.accuracy, feed_dict={self.input: X, self.output: Y, self.keep_prob: 1.0})


