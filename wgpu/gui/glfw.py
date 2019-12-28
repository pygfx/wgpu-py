"""
Support to render in a glfw window. The advantage of glfw is that it's
very lightweight.
"""

import ctypes

import glfw

from .base import BaseCanvas


# Expose GetWin32Window
# see https://github.com/FlorianRhiem/pyGLFW/issues/39
# see https://www.glfw.org/docs/latest/group__native.html
glfw._glfw.glfwGetWin32Window.restype = ctypes.c_void_p
glfw._glfw.glfwGetWin32Window.argtypes = [ctypes.POINTER(glfw._GLFWwindow)]
# todo: also for Linux and OS X


class WgpuCanvas(BaseCanvas):
    """ A canvas object wrapping a glfw window.
    """

    def __init__(self, *, size=None, title=None):
        if size:
            width, height = size
        else:
            width, height = 256, 256
        title = title or ""

        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        glfw.window_hint(glfw.RESIZABLE, True)
        self._window = glfw.create_window(width, height, title, None, None)
        glfw.set_window_refresh_callback(self._window, self._paint)

    def getSizeAndPixelRatio(self):
        # todo: maybe expose both logical size and physical size instead?
        width, height = glfw.get_window_size(self._window)
        pixelratio = glfw.get_window_content_scale(self._window)[0]
        return width, height, pixelratio

    def getWindowId(self):
        return int(glfw._glfw.glfwGetWin32Window(self._window))

    def _paint(self, *args):
        self._drawFrameAndPresent()

    def isClosed(self):
        return glfw.window_should_close(self._window)
