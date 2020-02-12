"""
Support to render in a glfw window. The advantage of glfw is that it's
very lightweight.

Install pyGLFW using ``pip install glfw``. On Windows this is enough.
On Linux, install the glfw lib using ``sudo apt install libglfw3``,
or ``sudo apt install libglfw3-wayland`` when using Wayland.
"""

import os
import sys
import ctypes

import glfw

from .base import BaseCanvas


glfw_version_info = tuple(int(i) for i in glfw.__version__.split(".")[:2])

if glfw_version_info < (1, 9):
    # todo: in half a year or so, remove this and force glfw >= 1.9
    # see https://github.com/FlorianRhiem/pyGLFW/issues/39
    # see https://www.glfw.org/docs/latest/group__native.html
    if hasattr(glfw._glfw, "glfwGetWin32Window"):
        glfw._glfw.glfwGetWin32Window.restype = ctypes.c_void_p
        glfw._glfw.glfwGetWin32Window.argtypes = [ctypes.POINTER(glfw._GLFWwindow)]
        glfw.get_win32_window = glfw._glfw.glfwGetWin32Window
    if hasattr(glfw._glfw, "glfwGetCocoaWindow"):
        glfw._glfw.glfwGetCocoaWindow.restype = ctypes.c_void_p
        glfw._glfw.glfwGetCocoaWindow.argtypes = [ctypes.POINTER(glfw._GLFWwindow)]
        glfw.get_cocoa_window = glfw._glfw.glfwGetCocoaWindow
    if hasattr(glfw._glfw, "glfwGetWaylandWindow"):
        glfw._glfw.glfwGetWaylandWindow.restype = ctypes.c_void_p
        glfw._glfw.glfwGetWaylandWindow.argtypes = [ctypes.POINTER(glfw._GLFWwindow)]
        glfw.get_wayland_window = glfw._glfw.glfwGetWaylandWindow
    if hasattr(glfw._glfw, "glfwGetWaylandDisplay"):
        glfw._glfw.glfwGetWaylandDisplay.restype = ctypes.c_void_p
        glfw._glfw.glfwGetWaylandDisplay.argtypes = []
        glfw.get_wayland_display = glfw._glfw.glfwGetWaylandDisplay
    if hasattr(glfw._glfw, "glfwGetX11Window"):
        glfw._glfw.glfwGetX11Window.restype = ctypes.c_uint32
        glfw._glfw.glfwGetX11Window.argtypes = [ctypes.POINTER(glfw._GLFWwindow)]
        glfw.get_x11_window = glfw._glfw.glfwGetX11Window
    if hasattr(glfw._glfw, "glfwGetX11Display"):
        glfw._glfw.glfwGetX11Display.restype = ctypes.c_void_p
        glfw._glfw.glfwGetX11Display.argtypes = []
        glfw.get_x11_display = glfw._glfw.glfwGetX11Display


# Do checks to prevent pitfalls on hybrid Xorg/Wayland systems
if sys.platform.startswith("linux"):
    is_wayland = "wayland" in os.getenv("XDG_SESSION_TYPE", "").lower()
    if is_wayland and not hasattr(glfw, "get_wayland_window"):
        raise RuntimeError(
            "We're on Wayland but Wayland functions not available. "
            + "Did you apt install libglfw3-wayland?"
        )


class GlfwWgpuCanvas(BaseCanvas):
    """ A canvas object wrapping a glfw window.
    """

    def __init__(self, *, size=None, title=None):
        super().__init__()
        if size:
            width, height = size
        else:
            width, height = 256, 256
        title = title or ""

        # Set window hints
        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        glfw.window_hint(glfw.RESIZABLE, True)
        # see https://github.com/FlorianRhiem/pyGLFW/issues/42
        # Alternatively, from pyGLFW 1.10 one can set glfw.ERROR_REPORTING='warn'
        if sys.platform.startswith("linux"):
            if "wayland" in os.getenv("XDG_SESSION_TYPE", "").lower():
                glfw.window_hint(glfw.FOCUSED, False)  # prevent Wayland focus error

        self._window = glfw.create_window(width, height, title, None, None)
        glfw.set_window_refresh_callback(self._window, self._paint)

    def get_size_and_pixel_ratio(self):
        # todo: maybe expose both logical size and physical size instead?
        width, height = glfw.get_window_size(self._window)
        pixelratio = glfw.get_window_content_scale(self._window)[0]
        return width, height, pixelratio

    def get_display_id(self):
        if sys.platform.startswith("linux"):
            is_wayland = "wayland" in os.getenv("XDG_SESSION_TYPE", "").lower()
            if is_wayland:
                return glfw.get_wayland_display()
            else:
                return glfw.get_x11_display()
        else:
            raise RuntimeError(f"Cannot get GLFW display id on {sys.platform}.")

    def get_window_id(self):
        if sys.platform.startswith("win"):
            return int(glfw.get_win32_window(self._window))
        elif sys.platform.startswith("darwin"):
            return int(glfw.get_cocoa_window(self._window))
        elif sys.platform.startswith("linux"):
            is_wayland = "wayland" in os.getenv("XDG_SESSION_TYPE", "").lower()
            if is_wayland:
                return int(glfw.get_wayland_window(self._window))
            else:
                return int(glfw.get_x11_window(self._window))
        else:
            raise RuntimeError(f"Cannot get GLFW window id on {sys.platform}.")

    def _paint(self, *args):
        self._draw_frame_and_present()

    def is_closed(self):
        return glfw.window_should_close(self._window)


WgpuCanvas = GlfwWgpuCanvas
