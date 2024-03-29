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
    """Minimal canvas interface implementation to support GPUCanvasContext"""

    def __init__(self, window):
        self._window = window

    def get_surface_info(self):
        # get window and display id
        # includes triage based on OS
        return get_surface_info(self._window)

    def get_physical_size(self):
        # get framebuffer size in integer pixels
        return get_physical_size(self._window)


def main():
    # get the gpu device/adapter combo
    device = get_default_device()

    # create a window with glfw
    glfw.init()
    # disable automatic API selection, we are not using opengl
    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
    glfw.window_hint(glfw.RESIZABLE, True)
    window = glfw.create_window(640, 480, "glfw window", None, None)

    # create a WGPU context
    canvas = GlfwCanvas(window)
    context = GPUCanvasContext(canvas)

    # drawing logic
    draw_frame = setup_draw(context, device)

    # render loop
    while True:
        # draw a frame
        draw_frame()
        # present the frame to the screen
        context.present()
        # process inputs
        glfw.poll_events()

        # break on close
        if glfw.window_should_close(window):
            break

    # dispose all resources and quit
    glfw.terminate()


if __name__ == "__main__":
    main()
