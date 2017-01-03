from preferentialnet.Network import *
from utils.Utils import *

if __name__ == "__main__":
    params = {
        'm': 20,
        'maxM': 1000
    }
    net = Network(params)
    degs = net.get_node_degs()
    Utils.plot_histogram(degs, 50)
