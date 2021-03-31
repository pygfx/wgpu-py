"""
Support to render in a glfw window. The advantage of glfw is that it's
very lightweight.

Install pyGLFW using ``pip install glfw``. On Windows this is enough.
On Linux, install the glfw lib using ``sudo apt install libglfw3``,
or ``sudo apt install libglfw3-wayland`` when using Wayland.
"""

import os
import sys
import weakref

import glfw

from .base import WgpuCanvasBase


# Make sure that glfw is new enough
glfw_version_info = tuple(int(i) for i in glfw.__version__.split(".")[:2])
if glfw_version_info < (1, 9):
    raise ImportError("wgpu-py requires glfw 1.9 or higher.")

# Do checks to prevent pitfalls on hybrid Xorg/Wayland systems
is_wayland = False
if sys.platform.startswith("linux"):
    is_wayland = "wayland" in os.getenv("XDG_SESSION_TYPE", "").lower()
    if is_wayland and not hasattr(glfw, "get_wayland_window"):
        raise RuntimeError(
            "We're on Wayland but Wayland functions not available. "
            + "Did you apt install libglfw3-wayland?"
        )


all_glfw_canvases = weakref.WeakSet()


def update_glfw_canvasses():
    """Call this in your glfw event loop to draw each canvas that needs
    an update. Returns the number of visible canvases.
    """
    # Note that _draw_frame_and_present already catches errors, it can
    # only raise errors if the logging system fails.
    canvases = tuple(all_glfw_canvases)
    for canvas in canvases:
        if canvas._need_draw:
            canvas._need_draw = False
            canvas._draw_frame_and_present()
    return len(canvases)


class GlfwWgpuCanvas(WgpuCanvasBase):
    """A glfw window providing a wgpu canvas."""

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
            if is_wayland:
                glfw.window_hint(glfw.FOCUSED, False)  # prevent Wayland focus error

        # Create the window (the initial size may not be in logical pixels)
        self._window = glfw.create_window(int(size[0]), int(size[1]), title, None, None)

        # Register ourselves
        self._need_draw = True
        self._changing_pixel_ratio = False
        all_glfw_canvases.add(self)

        # Register callbacks. We may get notified too often, but that's
        # ok, they'll result in a single draw.
        glfw.set_window_content_scale_callback(self._window, self._on_pixelratio_change)
        glfw.set_framebuffer_size_callback(self._window, self._on_size_change)
        glfw.set_window_close_callback(self._window, self._on_close)
        glfw.set_window_refresh_callback(self._window, self._on_window_dirty)
        glfw.set_window_focus_callback(self._window, self._on_window_dirty)
        glfw.set_window_maximize_callback(self._window, self._on_window_dirty)
        # Initialize the size
        self.set_logical_size(*size)

    # Callbacks

    def _on_pixelratio_change(self, *args):
        if self._changing_pixel_ratio:
            return
        self._changing_pixel_ratio = True  # prevent recursion (on Wayland)
        try:
            self._set_logical_size()
        finally:
            self._changing_pixel_ratio = False
        self._need_draw = True

    def _on_size_change(self, *args):
        self._logical_size = self._get_logical_size()
        self._need_draw = True

    def _on_close(self, *args):
        all_glfw_canvases.discard(self)
        glfw.hide_window(self._window)

    def _on_window_dirty(self, *args):
        self._need_draw = True

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
        # Apply
        if is_wayland:
            # Not sure why, but on Wayland things work differently
            screen_ratio = ssize[0] / lsize[0]
            glfw.set_window_size(
                self._window,
                int(lsize[0] / screen_ratio),
                int(lsize[1] / screen_ratio),
            )
        else:
            screen_ratio = ssize[0] / psize[0]
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
            if is_wayland:
                return int(glfw.get_wayland_window(self._window))
            else:
                return int(glfw.get_x11_window(self._window))
        else:
            raise RuntimeError(f"Cannot get GLFW window id on {sys.platform}.")

    def get_display_id(self):
        if sys.platform.startswith("linux"):
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

    def _request_draw(self):
        self._need_draw = True
        glfw.post_empty_event()  # Awake the event loop, if it's in wait-mode

    def close(self):
        glfw.set_window_should_close(self._window, True)
        self._on_close()

    def is_closed(self):
        return glfw.window_should_close(self._window)


# Make available under a name that is the same for all gui backends
WgpuCanvas = GlfwWgpuCanvas
