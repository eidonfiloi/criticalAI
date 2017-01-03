

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
