import unittest
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import tensorflow as tf
from preferentialnet.Network import *


class TFTest(unittest.TestCase):

    def test_histogram(self):
        x = np.random.rand(1000).astype(np.float32)

        # the histogram of the data
        n, bins, patches = plt.hist(x, 50, normed=1, facecolor='green', alpha=0.75)

        # add a 'best fit' line
        l = plt.plot(bins)

        plt.xlabel('Degree')
        plt.ylabel('Probability')
        plt.title(r'$\mathrm{Histogram\ of\ network degree distribution:}$')
        plt.axis([0, 1.0, 0, 1.0])
        plt.grid(True)

        plt.show()

    def test_sparse_matrix_creation(self):
        # row = [0, 0, 1, 2, 2, 2]
        #
        # col = [0, 2, 2, 0, 1, 2]
        # data = [1, 2, 3, 4, 5, 6]
        # csr_matrix((data, (row, col)), shape=(3, 3)).todense()

        params = {
            'm': 5,
            'maxM': 50
        }
        test_net = Network(params)
        weight_m = test_net.get_weight_matrix()

        print(weight_m)