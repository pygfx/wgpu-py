"""
Direct integration of glfw and wgpu-py without using the RenderCanvas library.

Demonstration for hardcore users that need total low-level control.

"""

# run_example = false

import os
import sys
import time
import atexit

import glfw
from wgpu.backends.wgpu_native import GPUCanvasContext

# from triangle import setup_drawing_sync
from cube import setup_drawing_sync


system_is_wayland = "wayland" in os.getenv("XDG_SESSION_TYPE", "").lower()
api_is_wayland = False
if sys.platform.startswith("linux") and system_is_wayland:
    if not hasattr(glfw, "get_x11_window"):
        api_is_wayland = True


def get_glfw_present_methods(window):
    if sys.platform.startswith("win"):
        return {
            "screen": {
                "platform": "windows",
                "window": int(glfw.get_win32_window(window)),
            }
        }
    elif sys.platform.startswith("darwin"):
        return {
            "screen": {
                "platform": "cocoa",
                "window": int(glfw.get_cocoa_window(window)),
            }
        }
    elif sys.platform.startswith("linux"):
        if api_is_wayland:
            return {
                "screen": {
                    "platform": "wayland",
                    "window": int(glfw.get_wayland_window(window)),
                    "display": int(glfw.get_wayland_display()),
                }
            }
        else:
            return {
                "screen": {
                    "platform": "x11",
                    "window": int(glfw.get_x11_window(window)),
                    "display": int(glfw.get_x11_display()),
                }
            }
    else:
        raise RuntimeError(f"Cannot get GLFW surface info on {sys.platform}.")


# Setup glfw
glfw.init()
atexit.register(glfw.terminate)


class MinimalGlfwCanvas:  # implements WgpuCanvasInterface
    """Minimal canvas interface required by wgpu."""

    def __init__(self, title):
        # disable automatic API selection, we are not using opengl
        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        glfw.window_hint(glfw.RESIZABLE, True)

        self.window = glfw.create_window(640, 480, title, None, None)
        self.context = GPUCanvasContext(self, get_glfw_present_methods(self.window))

    def get_physical_size(self):
        """get framebuffer size in integer pixels"""
        psize = glfw.get_framebuffer_size(self.window)
        return int(psize[0]), int(psize[1])

    def get_context(self, kind="wgpu"):
        return self.context


def main():
    # create canvas
    canvas = MinimalGlfwCanvas("wgpu gui direct")
    draw_frame = setup_drawing_sync(canvas)

    last_frame_time = time.perf_counter()
    frame_count = 0

    # render loop
    while not glfw.window_should_close(canvas.window):
        # process inputs
        glfw.poll_events()
        # draw a frame
        draw_frame()
        # present the frame to the screen
        canvas.context.present()
        # stats
        frame_count += 1
        etime = time.perf_counter() - last_frame_time
        if etime > 1:
            print(f"{frame_count / etime:0.1f} FPS")
            last_frame_time, frame_count = time.perf_counter(), 0

    # dispose resources
    glfw.destroy_window(canvas.window)

    # allow proper cleanup (workaround for glfw bug)
    end_time = time.perf_counter() + 0.1
    while time.perf_counter() < end_time:
        glfw.wait_events_timeout(end_time - time.perf_counter())


if __name__ == "__main__":
    main()
