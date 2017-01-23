from preferentialnet.Node import *
from preferentialnet.Link import *
from random import *
from copy import copy
import scipy.sparse

import numpy as np
import math


class PNetwork(object):
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

    def get_weight_matrix(self, shape=None):

        rows = []
        cols = []
        shape_n = len(self.nodes)
        for n in self.nodes:
            for l in n.links:
                rows.append(l.source.id)
                cols.append(l.target.id)
                rows.append(l.target.id)
                cols.append(l.source.id)
        size = len(cols)
        data = [1 for i in range(size)]
        matrix = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(shape_n, shape_n)).todense()
        matrix[matrix > 0] = 1

        if shape is not None:
            return matrix[:, np.random.randint(matrix.shape[0], size=shape)]
        else:
            return matrix


class LayerwisePNetwork(object):
    """
    This class represents a layered preferential network
    """

    def __init__(self, params):

        self.params = params
        self.layer_dim = self.params['layers']
        # number of start nodes and number of growing
        self.ms = self.params['m']
        self.in_degrees = [0 for i in range(len(self.layer_dim))]
        self.out_degrees = [0 for i in range(len(self.layer_dim))]
        self.layers = []
        self.nodes = []
        self.links = []

        # add start nodes on each layer
        for j in range(len(self.layer_dim)):
            node_id = 0
            nodes = []
            for i in range(self.ms[j]):
                node = PNode(node_id, j)
                nodes.append(node)
                self.nodes.append(node)
                node_id += 1
            self.layers.append(nodes)

        # add random start links between nodes layerwise
        for i in range(len(self.layer_dim) - 1):
            lenN = len(self.layers[i])
            for j in range(lenN):
                for k in range(lenN):
                    if random() < 0.5:
                        link1 = Link(self.layers[i][j], self.layers[i][k], 1.0)
                        link2 = Link(self.layers[i][k], self.layers[i][j], 1.0)
                        self.links.append(link1)
                        self.links.append(link2)
                        self.layers[i][j].out_links.append(link1)
                        self.layers[i][k].in_links.append(link1)
                        self.layers[i][j].in_links.append(link2)
                        self.layers[i][k].out_links.append(link2)
                        self.out_degrees[i] += 1
                        self.in_degrees[i] += 1
        self.grow_all_layers()

    def grow_all_layers(self):

        for i in range(len(self.layer_dim) - 1):
            lenN = len(self.layers[i])

            while lenN < self.layer_dim[i]:
                self.grow_step(i)
                lenN += 1
                print("layer {0} grew {1} nodes".format(i, lenN))

    def grow_step(self, layer):

        node = PNode(len(self.layers[layer]), layer)
        allEdgeWeights = self.out_degrees[layer] + self.in_degrees[layer] + 2 * len(self.layers[layer])
        i = 0
        newLinks = []
        while i < self.ms[layer]:
            randomTarget = np.random.choice(self.layers[layer], 1)[0]
            if random() < (randomTarget.sum_link_weights("all") / allEdgeWeights):
                link1 = Link(node, randomTarget, 1.0)
                link2 = Link(randomTarget, node, 1.0)
                newLinks.append(link1)
                newLinks.append(link2)
                randomTarget.in_links.append(link1)
                randomTarget.out_links.append(link2)
                node.out_links.append(link1)
                node.in_links.append(link2)
                self.in_degrees[layer] += 1
                self.out_degrees[layer] += 1
                i += 1

        self.layers[layer].append(node)
        self.nodes.append(node)
        for l in newLinks:
            self.links.append(l)

    def get_node_degs(self, layer, in_out):
        return [n.sum_link_weights(in_out) for n in self.layers[layer]]

    def get_all_weight_matrices(self):

        matrices = []
        for i in range(len(self.layer_dim) - 1):
            mat = self.get_weight_matrix(i, self.layer_dim[i+1])
            matrices.append(mat.astype('float32'))

        return matrices

    def get_weight_matrix(self, layer, shape=None):

        rows = []
        cols = []
        shape_n = len(self.layers[layer])
        for n in self.layers[layer]:
            for l_in in n.in_links:
                rows.append(l_in.source.id)
                cols.append(l_in.target.id)
            for l_out in n.out_links:
                rows.append(l_out.source.id)
                cols.append(l_out.target.id)
        size = len(cols)
        data = [1 for _ in range(size)]
        matrix = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(shape_n, shape_n)).todense()
        matrix[matrix > 0] = 1

        if shape is not None:
            return matrix[:, np.random.randint(matrix.shape[0], size=shape)]
        else:
            return matrix

    #     lens = [len(self.layers[i]) for i in range(len(self.layers))]
    #
    #     while True in [l1 < l2 for (l1, l2) in zip(lens, self.layer_dim)]:
    #         for i in range(len(lens)):
    #             if lens[i] < self.layer_dim[i] and i == 0:
    #                 node = PNode(self.last_node_id + 1, i)
    #                 self.grow_step_new(node, i, i + 1)
    #                 self.layers[i].append(node)
    #                 self.nodes.append(node)
    #                 self.last_node_id += 1
    #                 lens[i] += 1
    #                 print("layer {0} nodes are {1}".format(i, lens[i]))
    #             elif lens[i] < self.layer_dim[i] and i == len(lens) - 1:
    #                 node = PNode(self.last_node_id + 1, i)
    #                 self.grow_step_new(node, i, i - 1)
    #                 self.layers[i].append(node)
    #                 self.nodes.append(node)
    #                 self.last_node_id += 1
    #                 lens[i] += 1
    #                 print("layer {0} nodes are {1}".format(i, lens[i]))
    #             elif lens[i] < self.layer_dim[i] and 0 < i < len(lens) - 1:
    #                 node = PNode(self.last_node_id + 1, i)
    #                 self.grow_step_new(node, i, i-1)
    #                 self.grow_step_new(node, i, i+1)
    #                 self.layers[i].append(node)
    #                 self.nodes.append(node)
    #                 self.last_node_id += 1
    #                 lens[i] += 1
    #                 print("layer {0} nodes are {1}".format(i, lens[i]))
    #
    # def grow_step_new(self, node, layer_from, layer_to):
    #     allEdgeWeights = self.out_degrees[layer_to] + self.in_degrees[layer_to] + 2 * len(self.layers[layer_to])
    #     newLinks = []
    #     i = 0
    #     if layer_from > layer_to:
    #         while i < self.ms[layer_from]:
    #             randomTarget = np.random.choice(self.layers[layer_to], 1)[0]
    #             if random() < ((randomTarget.sum_link_weights("all") + 1) / allEdgeWeights):
    #                 link = Link(randomTarget, node, 1.0)
    #                 newLinks.append(link)
    #                 randomTarget.out_links.append(link)
    #                 self.out_degrees[layer_to] += 1
    #                 self.in_degrees[layer_from] += 1
    #                 node.in_links.append(link)
    #                 i += 1
    #
    #     else:
    #         while i < self.ms[layer_from]:
    #             randomTarget = np.random.choice(self.layers[layer_to], 1)[0]
    #             if random() < ((randomTarget.sum_link_weights("all") + 1) / allEdgeWeights):
    #                 link = Link(node, randomTarget, 1.0)
    #                 newLinks.append(link)
    #                 randomTarget.in_links.append(link)
    #                 self.in_degrees[layer_to] += 1
    #                 self.out_degrees[layer_from] += 1
    #                 node.out_links.append(link)
    #                 i += 1
    #     for l in newLinks:
    #         self.links.append(l)
    #
    # def grow_layers(self):
    #     # first grow first 2 layers
    #     print("grow first layer pair")
    #     lenN = len(self.layers[0])
    #     lenN1 = len(self.layers[1])
    #     while lenN < self.layer_dim[0] or lenN1 < self.layer_dim[1]:
    #         if lenN < self.layer_dim[0]:
    #             self.grow_step(0, "out")
    #             lenN += 1
    #             print("layer 0 nodes are {0}".format(lenN))
    #         if lenN1 < self.layer_dim[1]:
    #             self.grow_step(0, "in")
    #             lenN1 += 1
    #             print("layer 1 nodes are {0}".format(lenN1))
    #
    #     print("grow subsequent layer pairs")
    #     # grow subsequent layers by adding only in links to next layer
    #     for i in range(1, len(self.layer_dim) - 1):
    #         lenNN = len(self.layers[i+1])
    #         while lenNN < self.layer_dim[i+1]:
    #             self.grow_step(i, "in")
    #             print("layer {0} nodes are {1}".format(i + 1, lenNN))
    #             lenNN += 1
    #
    # def grow_step(self, layer_n, type_direction):
    #
    #     if type_direction == 'out':
    #         node = PNode(self.last_node_id + 1, layer_n)
    #         # allEdgeWeights = sum([sum(l.weight for l in n.in_links) for n in self.layers[layer_n + 1]])
    #         allEdgeWeights = self.in_degrees[layer_n + 1] + self.out_degrees[layer_n + 1] + 2*len(self.layers[layer_n + 1])
    #         i = 0
    #         newLinks = []
    #         while i < self.ms[layer_n]:
    #             randomTarget = np.random.choice(self.layers[layer_n + 1], 1)[0]
    #             if random() < ((randomTarget.sum_link_weights("all") + 1) / allEdgeWeights):
    #                 link = Link(node, randomTarget, 1.0)
    #                 newLinks.append(link)
    #                 randomTarget.in_links.append(link)
    #                 self.in_degrees[layer_n + 1] += 1
    #                 self.out_degrees[layer_n] += 1
    #                 node.out_links.append(link)
    #                 i += 1
    #         self.layers[layer_n].append(node)
    #         self.nodes.append(node)
    #         for l in newLinks:
    #             self.links.append(l)
    #         self.last_node_id += 1
    #
    #     elif type_direction == 'in':
    #         node = PNode(self.last_node_id + 1, layer_n + 1)
    #         # allEdgeWeights = sum([sum(l.weight for l in n.out_links) for n in self.layers[layer_n]])
    #         allEdgeWeights = self.out_degrees[layer_n] + self.in_degrees[layer_n] + 2*len(self.layers[layer_n])
    #         i = 0
    #         newLinks = []
    #         while i < self.ms[layer_n + 1]:
    #             randomTarget = np.random.choice(self.layers[layer_n], 1)[0]
    #             if random() < ((randomTarget.sum_link_weights("all") + 1) / allEdgeWeights):
    #                 link = Link(randomTarget, node, 1.0)
    #                 newLinks.append(link)
    #                 randomTarget.out_links.append(link)
    #                 self.out_degrees[layer_n] += 1
    #                 self.in_degrees[layer_n + 1] += 1
    #                 node.in_links.append(link)
    #                 i += 1
    #         self.layers[layer_n + 1].append(node)
    #         self.nodes.append(node)
    #         for l in newLinks:
    #             self.links.append(l)
    #         self.last_node_id += 1
    #
    #     else:
    #         node = PNode(self.last_node_id + 1, layer_n)
    #         # allEdgeWeights = sum([sum(l.weight for l in n.out_links) for n in self.layers[layer_n]])
    #         allEdgeWeights = self.out_degrees[layer_n] + self.in_degrees[layer_n] + 2 * len(self.layers[layer_n])
    #         i = 0
    #         newLinks = []
    #         while i < self.ms[layer_n + 1]:
    #             randomTarget = np.random.choice(self.layers[layer_n], 1)[0]
    #             if random() < ((randomTarget.sum_link_weights("all") + 1) / allEdgeWeights):
    #                 link = Link(randomTarget, node, 1.0)
    #                 newLinks.append(link)
    #                 randomTarget.out_links.append(link)
    #                 self.out_degrees[layer_n] += 1
    #                 self.in_degrees[layer_n + 1] += 1
    #                 node.in_links.append(link)
    #                 i += 1
    #         self.layers[layer_n + 1].append(node)
    #         self.nodes.append(node)
    #         for l in newLinks:
    #             self.links.append(l)
    #         self.last_node_id += 1

    def get_node_degs(self, layer_n, type):
        return [n.sum_link_weights(type) for n in self.layers[layer_n]]










