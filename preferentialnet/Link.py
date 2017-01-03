

class Link(object):
    """
    Class representing a link
    """

    def __init__(self, source, target, weight):
        self.id = "{0}-{1}".format(source.id, target.id)
        self.source = source
        self.target = target
        self.weight = weight
