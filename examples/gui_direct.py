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

# disable automatic API selection, we are not using opengl
glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
glfw.window_hint(glfw.RESIZABLE, True)


title = "wgpu glfw direct"
window = glfw.create_window(640, 480, title, None, None)
present_methods = get_glfw_present_methods(window)

context = GPUCanvasContext(present_methods)

# Initialize physical size once. For robust apps update this on resize events.
context.set_physical_size(glfw.get_framebuffer_size(window))


def main():
    # create canvas/context — setup_drawing_sync expects a "canvas" that can
    # provide get_context/get_physical_size; GPUCanvasContext now implements those.
    draw_frame = setup_drawing_sync(context)

    last_frame_time = time.perf_counter()
    frame_count = 0

    # render loop
    while not glfw.window_should_close(window):
        # process inputs
        glfw.poll_events()

        # resize handling
        context.set_physical_size(glfw.get_framebuffer_size(window))

        # draw a frame
        draw_frame()
        # present the frame to the screen
        context.present()
        # stats
        frame_count += 1
        etime = time.perf_counter() - last_frame_time
        if etime > 1:
            print(f"{frame_count / etime:0.1f} FPS")
            last_frame_time, frame_count = time.perf_counter(), 0

    # dispose resources
    glfw.destroy_window(window)

    # allow proper cleanup (workaround for glfw bug)
    end_time = time.perf_counter() + 0.1
    while time.perf_counter() < end_time:
        glfw.wait_events_timeout(end_time - time.perf_counter())


if __name__ == "__main__":
    main()
