import unittest
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import tensorflow as tf
from preferentialnet.PNetwork import *
import pickle
from utils.Utils import *
class TFTest(unittest.TestCase):

    def test_histogram(self):
        x = np.random.rand(1000).astype(np.float32)

        # the histogram of the raw_data
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
        # raw_data = [1, 2, 3, 4, 5, 6]
        # csr_matrix((raw_data, (row, col)), shape=(3, 3)).todense()

        params = {
            'm': 5,
            'maxM': 50
        }
        test_net = PNetwork(params)
        weight_m = test_net.get_weight_matrix()

        print(weight_m)

    def test_nonzero(self):

        d = {1: 2, 2: 1, 3: 3, 4: 1, 5: 2}

        s = sorted(set(d.values()), reverse=True)[:2]

        arr = np.array([1, 2, 0, 3, 0, 5])

        nz = tuple(np.nonzero(arr)[0])
        print(nz)

    def power_law_test(self):

        with open("../raw_data/frequent_activations_top1000_counts_ffnet_300_100.pickle", 'rb') as df:
            data = pickle.load(df)

            for j, d in enumerate(data):
                l = len(d)
                all_d = float(sum(d))
                # logrank = [math.log(i) for i in range(1, l+1)]
                logrank = [i for i in range(1, l+1)]
                logd = [math.log(float(el)/all_d) for el in d]
                # Utils.plot_line([logrank, logd], save_to="../raw_data/frequent_activations_top1000_counts_ffnet_300_100_layer_{0}".format(j))
                Utils.plot_line([logrank, logd])