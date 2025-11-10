import os
import sys


def get_glfw_present_info(window, vsync=True) -> dict:
    """Get the ``present_info`` dict required to instantiate a ``GPUCanvasContext``.

    Given a glfw window handle, return a dict that can be passed to ``wgpu.gpu.get_canvas_context()`` to create a ``GPUCanvasContext``.
    """

    # Lazy import, so the docs can be build without needing glfw
    import glfw

    system_is_wayland = "wayland" in os.getenv("XDG_SESSION_TYPE", "").lower()
    api_is_wayland = False
    if sys.platform.startswith("linux") and system_is_wayland:
        if not hasattr(glfw, "get_x11_window"):
            api_is_wayland = True

    if sys.platform.startswith("win"):
        return {
            "platform": "windows",
            "window": int(glfw.get_win32_window(window)),
            "vsync": vsync,
        }
    elif sys.platform.startswith("darwin"):
        return {
            "platform": "cocoa",
            "window": int(glfw.get_cocoa_window(window)),
            "vsync": vsync,
        }
    elif sys.platform.startswith("linux"):
        if api_is_wayland:
            return {
                "platform": "wayland",
                "window": int(glfw.get_wayland_window(window)),
                "display": int(glfw.get_wayland_display()),
                "vsync": vsync,
            }
        else:
            return {
                "platform": "x11",
                "window": int(glfw.get_x11_window(window)),
                "display": int(glfw.get_x11_display()),
                "vsync": vsync,
            }
    else:
        raise RuntimeError(f"Cannot get GLFW surface info on {sys.platform}.")
