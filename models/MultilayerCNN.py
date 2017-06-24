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
        self.input_image_shape = self.params['input_image_shape']
        self.layer_params = self.params['layer_params']
        self.batch_size = self.params['batch_size']
        self.num_classes = self.params['num_classes']

        self.tensorboard_dir = self.params['tensorboard_dir']

        self.activation_function = self.params['activation_function']
        self.optimizer_ = self.params['optimizer']
        # model
        self.input = tf.placeholder(tf.float32, [None, self.input_dim])
        self.output = tf.placeholder(tf.float32, [None, self.output_dim])

        self.layer_output = tf.reshape(self.input, self.input_image_shape)

        self.activation_patterns = {}
        for layer, p in enumerate(self.layer_params):
            with tf.name_scope("layer_{0}".format(layer)):
                if p['type'] == 'conv':
                    with tf.name_scope("conv"):
                        W_conv = tf.Variable(tf.truncated_normal(shape=p["W_shape"], stddev=0.01), name="W_conv")
                        b_conv = tf.Variable(tf.constant(0.01, shape=p["b_shape"]), name="b_conv")
                        Utils.variable_summaries(W_conv, "W_conv")
                        Utils.variable_summaries(b_conv, "b_conv")

                        h_conv = self.activation_function(tf.nn.conv2d(self.layer_output, W_conv, strides=[1, 1, 1, 1], padding='SAME') + b_conv, name="conv")

                        Utils.variable_summaries(h_conv, "h_conv")
                        h_norm = tf.nn.lrn(h_conv, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
                        h_pool = tf.nn.max_pool(h_norm, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool")
                        self.layer_output = h_pool
                elif p['type'] == 'local':
                    with tf.name_scope("local"):
                        if p['first']:
                            reshape = tf.reshape(self.layer_output, [-1, p["W_shape"][0]])
                            weights = tf.Variable(tf.truncated_normal(shape=p["W_shape"], stddev=0.01),
                                                  name="W_local")
                            biases = tf.Variable(tf.constant(0.01, shape=p["b_shape"]), name="b_local")
                            Utils.variable_summaries(weights, "W")
                            Utils.variable_summaries(biases, "W")
                            self.layer_output = self.activation_function(tf.matmul(reshape, weights) + biases,
                                                                         name="layer_output")
                        else:
                            weights = tf.Variable(tf.truncated_normal(shape=p["W_shape"], stddev=0.01),
                                                  name="W_local")
                            biases = tf.Variable(tf.constant(0.01, shape=p["b_shape"]), name="b_local")
                            self.layer_output = self.activation_function(tf.matmul(self.layer_output, weights) + biases,
                                                                         name="layer_output")
                            Utils.variable_summaries(weights, "W")
                            Utils.variable_summaries(biases, "W")
                        self.activation_patterns['layer_{0}'.format(layer)] = self.layer_output
                else:
                    with tf.name_scope('softmax_linear'):
                        weights = tf.Variable(tf.truncated_normal(shape=[p["W_shape"], self.num_classes], stddev=0.01),
                                              name="W")
                        biases = tf.Variable(tf.constant(0.01, shape=[self.num_classes]), name="b")
                        self.layer_output = self.activation_function(tf.matmul(self.layer_output, weights) + biases,
                                                                     name="layer_output")

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.layer_output, labels=self.output))
        tf.summary.scalar('cross_entropy', self.cross_entropy)
        self.train_step = self.optimizer_.minimize(self.cross_entropy)
        correct_prediction = tf.equal(tf.argmax(self.layer_output, 1), tf.argmax(self.output, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', self.accuracy)

        self.sess = tf.Session()
        self.merged = tf.summary.merge_all()
        self.summ_writer = tf.summary.FileWriter(self.tensorboard_dir, self.sess.graph)

        init = tf.global_variables_initializer()
        self.sess.run(init)

        # W_conv1 = Utils.weight_variable([5, 5, 1, 32])
        # b_conv1 = Utils.bias_variable([32])
        #
        # h_conv1 = self.activation_function(Utils.conv2d(x_image, W_conv1) + b_conv1)
        # h_pool1 = Utils.max_pool_2x2(h_conv1)
        #
        # W_conv2 = Utils.weight_variable([5, 5, 32, 64])
        # b_conv2 = Utils.bias_variable([64])
        #
        # h_conv2 = self.activation_function(Utils.conv2d(h_pool1, W_conv2) + b_conv2)
        # h_pool2 = Utils.max_pool_2x2(h_conv2)
        #
        # W_fc1 = Utils.weight_variable([7 * 7 * 64, 1024])
        # b_fc1 = Utils.bias_variable([1024])
        #
        # h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        # h_fc1 = self.activation_function(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        #
        # self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        # h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
        #
        # W_fc2 = Utils.weight_variable([1024, 10])
        # b_fc2 = Utils.bias_variable([10])
        #
        # y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        #
        # self.h_conv1_flat = tf.reshape(h_conv1, [-1, 14 * 14 * 32])
        # self.h_conv2_flat = tf.reshape(h_conv2, [-1, 7 * 7 * 64])
        # self.h_fc1 = h_fc1
        # self.h_fc2 = y_conv
        #
        # self.activation_dict = {'h_conv1': self.h_conv1_flat,
        #                         'h_conv2': self.h_conv2_flat,
        #                         'h_fc1': self.h_fc1,
        #                         'h_fc2': self.h_fc2}
        #
        # self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, self.output))
        # tf.summary.scalar('cross_entropy', self.cross_entropy)
        # self.train_step = self.optimizer_.minimize(self.cross_entropy)
        # correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(self.output, 1))
        # self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # tf.summary.scalar('accuracy', self.accuracy)
        #
        # self.sess = tf.Session()
        # self.merged = tf.summary.merge_all()
        # self.summ_writer = tf.train.SummaryWriter(self.tensorboard_dir, self.sess.graph)
        # init = tf.initialize_all_variables()
        # self.sess.run(init)

    def partial_fit(self, X, Y):
        cost, opt, merged = self.sess.run((self.cross_entropy, self.train_step, self.merged),
                                            feed_dict={self.input: X, self.output: Y})
        return cost, merged

    def get_activations(self, X):
        # return [self.sess.run(v, feed_dict={self.input: X}) for _, v in self.activation_patterns.items()]
        return {k: self.sess.run(v, feed_dict={self.input: X}) for k, v in self.activation_patterns.items()}

    def inference(self, X, Y):
        accuracy = self.sess.run(self.accuracy, feed_dict={self.input: X, self.output: Y})
        return accuracy

    def get_accuracy(self, X, Y):
        return self.sess.run(self.accuracy, feed_dict={self.input: X, self.output: Y})


