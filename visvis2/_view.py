from .objects._base import WorldObject


# todo: this is a WidgetObject


class View:
    """ Represents a square region inside a Figure. Each View has one
    scenegraph and one active camera.
    """

    def __init__(self):

        self._scene = WorldObject()

    @property
    def scene(self):
        return self._scene
