import ctypes
import asyncio

import glfw
import vulkan as vk

from ._surface import SurfaceBackend
from ._instance import register_standard_extension


glfw.init()


# Register the extensions that we need
for x in glfw.get_required_instance_extensions():
    register_standard_extension(str(x))


class GLFWSurfaceBackend(SurfaceBackend):
    """ Surface backend for GLFW.
    """

    def create_surface(self, instance_handle, width, height, name):

        # Create a window
        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        glfw.window_hint(glfw.RESIZABLE, False)
        window = glfw.create_window(width, height, name, None, None)
        self._window = window

        # Create the Vulkan surface object
        ffi = vk.ffi
        surface = ctypes.c_void_p(0)
        # instance = ctypes.cast(int(ffi.cast('intptr_t', instance_handle)), ctypes.c_void_p)
        instance = ctypes.cast(
            int(ffi.cast("uintptr_t", instance_handle)), ctypes.c_void_p
        )
        glfw.create_window_surface(instance, window, None, ctypes.byref(surface))
        surface = ffi.cast("VkSurfaceKHR", surface.value)
        if surface is None:
            raise Exception("failed to create window surface!")
        return surface

    def integrate_asyncio(self, loop):
        loop.create_task(self._keep_glfw_alive())

    async def _keep_glfw_alive(self):
        while True:
            await asyncio.sleep(0.1)
            if glfw.window_should_close(self._window):
                glfw.terminate()
                break
            else:
                glfw.poll_events()
