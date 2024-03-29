"""
Direct integration of glfw and wgpu-py without using the
wgpu.gui Canvas abstraction/class hierarchy.

Demonstration for hardcore users that need total low-level
control.

# run_example = false
"""

import sys
from pathlib import Path

import glfw

from wgpu.backends.wgpu_native import GPUCanvasContext
from wgpu.gui.glfw import get_surface_info, get_physical_size
from wgpu.utils.device import get_default_device


sys.path.insert(0, str(Path(__file__).parent))

from triangle import setup_draw  # noqa: E402


class GlfwCanvas:
    def __init__(self, window):
        self._window = window

    def get_surface_info(self):
        return get_surface_info(self._window)

    def get_physical_size(self):
        return get_physical_size(self._window)


def main():
    device = get_default_device()

    glfw.init()
    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
    glfw.window_hint(glfw.RESIZABLE, True)
    window = glfw.create_window(640, 480, "glfw window", None, None)

    canvas = GlfwCanvas(window)
    context = GPUCanvasContext(canvas)

    draw_frame = setup_draw(context, device)

    while True:
        draw_frame()
        context.present()
        glfw.poll_events()
        if glfw.window_should_close(window):
            break
    glfw.terminate()


if __name__ == "__main__":
    main()
