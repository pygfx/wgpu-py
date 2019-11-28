class WorldObject:
    """ The base class for objects present in the "world", i.e. the scene graph.
    """

    def __init__(self):

        self._children = []

    @property
    def children(self):
        return self._children
