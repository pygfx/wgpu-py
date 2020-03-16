"""
Support to render in a glfw window. The advantage of glfw is that it's
very lightweight.

Install pyGLFW using ``pip install glfw``. On Windows this is enough.
On Linux, install the glfw lib using ``sudo apt install libglfw3``,
or ``sudo apt install libglfw3-wayland`` when using Wayland.
"""

import os
import sys

import glfw

from .base import BaseCanvas


# Make sure that glfw is new enough
glfw_version_info = tuple(int(i) for i in glfw.__version__.split(".")[:2])
if glfw_version_info < (1, 9):
    raise ImportError("wgpu-py requires glfw 1.9 or higher.")

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

    # See https://www.glfw.org/docs/latest/group__window.html

    def __init__(self, *, size=None, title=None):
        super().__init__()

        # Handle inputs
        if not size:
            size = 640, 480
        title = str(title or "")

        # Set window hints
        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        glfw.window_hint(glfw.RESIZABLE, True)
        # see https://github.com/FlorianRhiem/pyGLFW/issues/42
        # Alternatively, from pyGLFW 1.10 one can set glfw.ERROR_REPORTING='warn'
        if sys.platform.startswith("linux"):
            if "wayland" in os.getenv("XDG_SESSION_TYPE", "").lower():
                glfw.window_hint(glfw.FOCUSED, False)  # prevent Wayland focus error

        # Create the window (the initial size may not be in logical pixels)
        self._window = glfw.create_window(int(size[0]), int(size[1]), title, None, None)

        # Register callbacks
        glfw.set_window_content_scale_callback(self._window, self._on_pixelratio_change)
        glfw.set_framebuffer_size_callback(self._window, self._on_size_change)
        glfw.set_window_refresh_callback(self._window, self._on_paint)
        if sys.platform.startswith("darwin"):
            # Apparently, the refresh_callback is not called when the window
            # is created, gets focus, or is resized on macOS. So this is a
            # workaround to explicitely make sure that the paint callback
            # is called so that the contents are drawn.
            glfw.set_window_focus_callback(self._window, self._paint)
        # glfw.set_window_iconify_callback
        # glfw.set_window_maximize_callback
        # glfw.set_framebuffer_size_callback

        # Initialize the size
        self.set_logical_size(*size)

    # Callbacks

    def _on_pixelratio_change(self, *args):
        self._set_logical_size()
        self._on_paint()

    def _on_size_change(self, *args):
        self._logical_size = self._get_logical_size()
        self._on_paint()

    def _on_paint(self, *args):
        self._draw_frame_and_present()

    # Helpers

    def _get_logical_size(self):
        # Because the value of get_window_size is in physical pixels
        # on some systems and logical pixels on other, we use the
        # framebuffer size and pixel ratio to derive the logical size.
        psize = glfw.get_framebuffer_size(self._window)
        psize = int(psize[0]), int(psize[1])
        ratio = glfw.get_window_content_scale(self._window)[0]
        return psize[0] / ratio, psize[1] / ratio

    def _set_logical_size(self):
        # There is unclarity about the window size in "screen pixels".
        # It appears that on Windows and X11 its the same as the
        # framebuffer size, and on macOS it's logical pixels.
        # See https://github.com/glfw/glfw/issues/845
        # Here, we simply do a quick test so we can compensate.

        # The target logical size
        lsize = self._logical_size
        pixel_ratio = glfw.get_window_content_scale(self._window)[0]
        # The current screen size and physical size, and its ratio
        ssize = glfw.get_window_size(self._window)
        psize = glfw.get_framebuffer_size(self._window)
        screen_ratio = ssize[0] / psize[0]
        # Apply
        glfw.set_window_size(
            self._window,
            int(lsize[0] * pixel_ratio / screen_ratio),
            int(lsize[1] * pixel_ratio / screen_ratio),
        )

    # API

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

    def get_display_id(self):
        if sys.platform.startswith("linux"):
            is_wayland = "wayland" in os.getenv("XDG_SESSION_TYPE", "").lower()
            if is_wayland:
                return glfw.get_wayland_display()
            else:
                return glfw.get_x11_display()
        else:
            raise RuntimeError(f"Cannot get GLFW display id on {sys.platform}.")

    def get_pixel_ratio(self):
        return glfw.get_window_content_scale(self._window)[0]

    def get_logical_size(self):
        return self._logical_size

    def get_physical_size(self):
        psize = glfw.get_framebuffer_size(self._window)
        return int(psize[0]), int(psize[1])

    def set_logical_size(self, width, height):
        if width < 0 or height < 0:
            raise ValueError("Window width and height must not be negative")
        self._logical_size = float(width), float(height)
        self._set_logical_size()

    def close(self):
        # glfw.hide_window(self._window) - no: clicking the cross also does not hide it
        glfw.set_window_should_close(self._window, True)

    def is_closed(self):
        return glfw.window_should_close(self._window)


WgpuCanvas = GlfwWgpuCanvas
