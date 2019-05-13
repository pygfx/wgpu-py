import asyncio

import vulkan as vk

from ._core import GPUObject


class Surface(GPUObject):
    """ Represents a screen surface to draw to.
    Needs a windowing toolkit (glfw only for now, but later also sdl2 and qt).
    """

    def __init__(self):
        self._instance = None
        self._loop = None

        # todo: Select backend
        self._backend = GLFWSurfaceBackend()

    def get_required_extensions(self):
        """ Get a list of Vulkan extensions required by the surface.
        """
        return self._backend.get_required_extensions()

    def _create_surface(self, instance):
        """ Create the actual surface. This is called by the Instance object.
        Cannot be done from ``__init__()`` because the instance needs the
        required extensions to initialize itself.
        """
        if self._instance is not None:
            raise RuntimeError("Surface already created.")
        self._instance = instance
        self._handle = self._backend.create_surface(
            instance._handle, 640, 480, "visvis2"
        )

    def _destroy(self):
        if self._handle and self._instance and self._instance._handle:
            func = vk.vkGetInstanceProcAddr(
                self._instance._handle, "vkDestroySurfaceKHR"
            )
            if func:
                func(self._instance._handle, self._handle, None)
        self._handle = None

    def integrate_asyncio(self, loop=None):
        if self._loop is not None:
            raise RuntimeError("Already integrated")
        if loop is None:
            loop = asyncio.get_event_loop()

        self._loop = loop
        self._backend.integrate_asyncio(loop)


class SurfaceBackend:
    pass


from ._surface_glfw import GLFWSurfaceBackend
