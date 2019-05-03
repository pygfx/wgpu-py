import asyncio

import vulkan as vk

from ._core import GPUObject


class Surface(GPUObject):

    def __init__(self, instance):
        self._instance = instance
        self._loop = None

        self._backend = GLFWSurfaceBackend()
        self._handle =self._backend.create_surface(instance._handle, 640, 480, "visvis2")

    def _destroy(self):
        if self._handle and self._instance._handle:
            func = vk.vkGetInstanceProcAddr(self._instance._handle, 'vkDestroySurfaceKHR')
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
