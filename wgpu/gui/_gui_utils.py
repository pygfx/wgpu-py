""" Private gui utilities.
"""

import os
import sys
import ctypes.util
import logging


logger = logging.getLogger("wgpu")


SYSTEM_IS_WAYLAND = "wayland" in os.getenv("XDG_SESSION_TYPE", "").lower()

if sys.platform.startswith("linux") and SYSTEM_IS_WAYLAND:
    # Force glfw to use X11. Note that this does not work if glfw is already imported.
    if "glfw" not in sys.modules:
        os.environ["PYGLFW_LIBRARY_VARIANT"] = "x11"
    # Force Qt to use X11. Qt is more flexible - it ok if e.g. PySide6 is already imported.
    os.environ["QT_QPA_PLATFORM"] = "xcb"


_x11_display = None


def get_alt_x11_display():
    """Get (the pointer to) a process-global x11 display instance."""
    # Ideally we'd get the real display object used by the GUI toolkit.
    # But this is not always possible. In that case, using an alt display
    # object can be used.
    global _x11_display
    assert sys.platform.startswith("linux")
    if _x11_display is None:
        x11 = ctypes.CDLL(ctypes.util.find_library("X11"))
        x11.XOpenDisplay.restype = ctypes.c_void_p
        _x11_display = x11.XOpenDisplay(None)
    return _x11_display


_wayland_display = None


def get_alt_wayland_display():
    """Get (the pointer to) a process-global Wayland display instance."""
    # Ideally we'd get the real display object used by the GUI toolkit.
    # This creates a global object, similar to what we do for X11.
    # Unfortunately, this segfaults, so it looks like the real display object
    # is needed? Leaving this here for reference.
    global _wayland_display
    assert sys.platform.startswith("linux")
    if _wayland_display is None:
        wl = ctypes.CDLL(ctypes.util.find_library("wayland-client"))
        wl.wl_display_connect.restype = ctypes.c_void_p
        _wayland_display = wl.wl_display_connect(None)
    return _wayland_display
