from preferentialnet.Node import *
from preferentialnet.Link import *
from random import *
from copy import copy
import scipy.sparse

import numpy as np

class Network(object):
    """
    Class representing a network
    """

    def __init__(self, params):

        self.params = params
        # number of start nodes and number of growing
        self.m = self.params['m'] if self.params['m'] is not None else 20
        self.maxM = self.params['maxM'] if self.params['maxM'] is not None else 500
        self.nodes = [Node(k) for k in range(self.m)]
        self.links = []

        # add random links between nodes
        for i in range(self.m):
            for j in range(self.m):
                if i < j and random() < 0.5:
                    link = Link(self.nodes[i], self.nodes[j], 1.0)
                    self.links.append(link)
                    self.nodes[i].links.append(link)
                    self.nodes[j].links.append(link)

        self.grow()

    def grow(self):
        while len(self.nodes) < self.maxM:
            if len(self.nodes) % 100 == 0:
                print("next grow step, nodes are {0}".format(len(self.nodes)))
            self.grow_step()

    def grow_step(self):

        node = Node(len(self.nodes))

        allEdgeWeights = sum([sum(l.weight for l in n.links) for n in self.nodes]) / 2
        i = 0
        newLinks = []
        while i < self.m:
            randomTarget = np.random.choice(self.nodes, 1)[0]
            if random() < (randomTarget.sum_link_weights() / allEdgeWeights):
                link = Link(node, randomTarget, 1.0)
                newLinks.append(link)
                randomTarget.links.append(link)
                node.links.append(link)
                i += 1

        self.nodes.append(node)
        for l in newLinks:
            self.links.append(l)

    def get_node_degs(self):
        return [n.sum_link_weights() for n in self.nodes]

    def get_weight_matrix(self):

        rows = []
        cols = []
        shape_n = len(self.nodes)
        for n in self.nodes:
            for l in n.links:
                rows.append(l.source.id)
                cols.append(l.target.id)
        size = len(cols)
        data = [1 for i in range(size)]
        matrix = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(shape_n, shape_n)).todense()
        matrix[matrix > 1] = 1

        return matrix











