from preferentialnet.PNetwork import *
from utils.Utils import *

if __name__ == "__main__":
    # params = {
    #     'm': 20,
    #     'maxM': 1000
    # }
    # net = PNetwork(params)
    # degs = net.get_node_degs()
    # Utils.plot_histogram(degs, 50)

    params = {
        'layers': [200, 100, 50, 10],
        'm': [10, 5, 5, 10]
    }

    net = LayerwisePNetwork(params)

    matrices = net.get_all_weight_matrices()

    degs = net.get_node_degs(0, 'out')
    Utils.plot_histogram(degs, 50)

    degs1out = net.get_node_degs(1, 'out')
    Utils.plot_histogram(degs1out, 50)

    degs1in = net.get_node_degs(1, 'in')
    Utils.plot_histogram(degs1in, 50)

    degs2out = net.get_node_degs(2, 'out')
    Utils.plot_histogram(degs1in, 50)

    degs2in = net.get_node_degs(2, 'in')
    Utils.plot_histogram(degs1in, 50)

