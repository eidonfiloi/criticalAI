import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


class Utils(object):
    """

    """

    @staticmethod
    def plot_histogram(data, binN):
        # the histogram of the data

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

