"""
Support to render in a glfw window. The advantage of glfw is that it's
very lightweight.
"""

import sys
import ctypes

import glfw

from .base import BaseCanvas


glfw_version_info = tuple(int(i) for i in glfw.__version__.split(".")[:2])

if glfw_version_info < (1, 9):
    # see https://github.com/FlorianRhiem/pyGLFW/issues/39
    # see https://www.glfw.org/docs/latest/group__native.html
    if sys.platform.startswith("win"):
        glfw._glfw.glfwGetWin32Window.restype = ctypes.c_void_p
        glfw._glfw.glfwGetWin32Window.argtypes = [ctypes.POINTER(glfw._GLFWwindow)]
        glfw.get_win32_window = glfw._glfw.glfwGetWin32Window  # todo: name ok?
    elif sys.platform.startswith("darwin"):
        glfw._glfw.glfwGetCocoaWindow.restype = ctypes.c_void_p
        glfw._glfw.glfwGetCocoaWindow.argtypes = [ctypes.POINTER(glfw._GLFWwindow)]
        glfw.get_cocoa_window = glfw._glfw.glfwGetCocoaWindow
    # todo: also for Linux


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
        if sys.platform.startswith("win"):
            return int(glfw.get_win32_window(self._window))
        elif sys.platform.startswith("darwin"):
            return int(glfw.get_cocoa_window(self._window))
        else:
            raise NotImplementedError()

    def _paint(self, *args):
        self._drawFrameAndPresent()

    def isClosed(self):
        return glfw.window_should_close(self._window)
