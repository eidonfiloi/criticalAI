

class Node(object):
    """
    Class representing a Node in the Network
    """

    def __init__(self, id_):
        self.id = id_
        self.links = []
        self.output = 0.0

    def sum_link_weights(self):
        return sum([l.weight for l in self.links])


class PNode(object):
    """
        Class representing a preferential layer Node in the Network
        """

    def __init__(self, id_, layer_id_=0):
        self.id = id_
        self.layer_id = layer_id_
        self.in_links = []
        self.out_links = []
        self.output = 0.0

    def sum_link_weights(self, in_out_all="in"):
        if in_out_all == "in":
            return sum([l.weight for l in self.in_links])
        elif in_out_all == "out":
            return sum([l.weight for l in self.out_links])
        else:
            return sum([l.weight for l in self.out_links]) + sum([l.weight for l in self.in_links])