import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import tensorflow as tf


class Utils(object):
    """

    """

    @staticmethod
    def plot_histogram(data, binN):
        # the histogram of the raw_data

        plt.figure()
        n, bins, patches = plt.hist(np.log2(data), binN, log=True, facecolor='green', alpha=0.75)

        # add a 'best fit' line
        # plt.plot(np.log10(bins))
        # plt.gca().set_xscale("log")
        # plt.gca().set_yscale("log")

        plt.xlabel('Degree')
        plt.ylabel('Probability')
        plt.title(r'$\mathrm{Histogram\ of\ network degree distribution:}$')
        # plt.axis([0, 200, 0, 40.0])
        plt.grid(True)

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
