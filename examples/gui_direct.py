"""
Direct integration of glfw and wgpu-py without using the
wgpu.gui Canvas abstraction/class hierarchy.

Demonstration for hardcore users that need total low-level control.

"""

# run_example = false

import time

import glfw

from wgpu.backends.wgpu_native import GPUCanvasContext
from wgpu.gui.glfw import get_glfw_present_info, get_physical_size

from triangle import setup_triangle  # noqa: F401, RUF100
from cube import setup_cube  # noqa: F401, RUF100


class MinimalGlfwCanvas:  # implements WgpuCanvasInterface
    """Minimal canvas interface implementation triangle.py has everything it needs to draw."""

    def __init__(self, window):
        self._window = window
        self.context = GPUCanvasContext(self)
        self.draw_frame = None

    def get_present_info(self):
        """get window and display id, includes some triage to deal with OS differences"""
        return get_glfw_present_info(self._window)

    def get_physical_size(self):
        """get framebuffer size in integer pixels"""
        return get_physical_size(self._window)

    def get_context(self, kind="webgpu"):
        return self.context

    def request_draw(self, func=None):
        # A method from WGPUCanvasBase that is called by triangle.py
        if func is not None:
            self.draw_frame = func


def main():
    # create a window with glfw
    glfw.init()
    # disable automatic API selection, we are not using opengl
    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
    glfw.window_hint(glfw.RESIZABLE, True)
    window = glfw.create_window(640, 480, "wgou demo glfw direct", None, None)

    # create canvas
    canvas = MinimalGlfwCanvas(window)
    setup_cube(canvas)

    last_frame_time = time.perf_counter()
    frame_count = 0

    # render loop
    while True:
        # process inputs
        glfw.poll_events()
        # break on close
        if glfw.window_should_close(window):
            break
        # draw a frame
        canvas.draw_frame()
        # present the frame to the screen
        canvas.context.present()
        # stats
        frame_count += 1
        etime = time.perf_counter() - last_frame_time
        if etime > 1:
            print(f"{frame_count/etime:0.1f} FPS")
            last_frame_time, frame_count = time.perf_counter(), 0

    # dispose all resources and quit
    glfw.destroy_window(window)
    glfw.poll_events()
    glfw.terminate()


if __name__ == "__main__":
    main()
