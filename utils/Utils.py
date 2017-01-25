import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import math


class Utils(object):
    """

    """

    @staticmethod
    def plot_histogram(data, title="", binN=30, save_to=None, loglog=True):
        # the histogram of the raw_data

        if loglog:
            data_ = np.log10(data)
        else:
            data_ = data
        cmap = plt.get_cmap('viridis')
        color = random.choice(cmap.colors)
        binN = 2*int(math.sqrt(len(data_)))
        binw = (max(data_) - min(data_)) / binN
        bins = np.arange(min(data_), max(data_) + binw, binw)

        plt.figure()
        if loglog:
            plt.hist(data_, bins=bins, log=True, color=color, alpha=0.90)
        else:
            plt.hist(data_, bins=bins, log=False, color=color, alpha=0.90)

        plt.xlabel('weight degree')
        plt.ylabel('Counts')
        plt.title(title)
        plt.grid(True)

        if save_to is not None:
            plt.savefig(save_to)
        else:
            plt.show()

    @staticmethod
    def plot_line(data, save_to=None):

        plt.figure()
        plt.plot(data[0], data[1], 'ro')

        plt.xlabel('log rank')
        plt.ylabel('log activation frequency')
        if save_to is None:
            plt.show()
        else:
            plt.savefig(save_to)

    @staticmethod
    def variable_summaries(var, name="summaries"):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization).
        :param name:
        """
        with tf.name_scope(name):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    @staticmethod
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')
