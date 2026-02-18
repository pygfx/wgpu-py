"""
GUI Direct 2
------------

Direct integration of glfw and wgpu-py without using the RenderCanvas library.

Demonstration for hardcore users that need total low-level control.

This version is a bit more elaborate, using a loop object, and allowing
async code that uses`promose.then()`. For this to work, wgpu needs
access to the running loop. In particular, it needs a
run_soon_threadsafe() function, allowing wgpu to resolve a promise from
its internal thread.
"""

# run_example = false

import sys
import time
import atexit

import glfw
import wgpu
from wgpu.utils.glfw_present_info import get_glfw_present_info

# from triangle import setup_drawing_sync
from cube import setup_drawing_sync

# Setup glfw
glfw.init()
atexit.register(glfw.terminate)

# disable automatic API selection, we are not using opengl
glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
glfw.window_hint(glfw.RESIZABLE, True)


title = "wgpu glfw direct"
window = glfw.create_window(640, 480, title, None, None)
present_info = get_glfw_present_info(window)

context = wgpu.gpu.get_canvas_context(present_info)

# Initialize physical size once. For robust apps update this on resize events.
context.set_physical_size(*glfw.get_framebuffer_size(window))


# Setup async callbacks. This is optional, but it enables code using promise.then().
# The asyncgen hook is a stub for the system to detect the call_soon_threadsafe function.
# This works if both are defined on the same class or in the same module.
to_call_soon = []
call_soon_threadsafe = to_call_soon.append
stub_asynchen_hook = lambda agen: None
sys.set_asyncgen_hooks(stub_asynchen_hook)


class Loop:
    def __init__(self):
        self._pending_callbacks = []

    def _asynchen_hook(self, agen):
        pass

    def call_soon_threadsafe(self, callback):
        self._pending_callbacks.append(callback)

    def run(self):
        # Setup async hook. Our stub hook does not do anything, but it makes that
        # the wgpu promises detect this loop and the call_soon_threadsafe method.
        sys.set_asyncgen_hooks(self._asynchen_hook)

        # Setup
        draw_frame = setup_drawing_sync(context)
        last_frame_time = time.perf_counter()
        frame_count = 0

        # render loop
        while not glfw.window_should_close(window):
            # process inputs
            glfw.poll_events()

            # resize handling
            context.set_physical_size(*glfw.get_framebuffer_size(window))

            # call async callbacks (optional, see above)
            while self._pending_callbacks:
                callback = self._pending_callbacks.pop(0)
                print("Callback:", callback)
                try:
                    callback()
                except Exception as err:
                    print(err)

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
        sys.set_asyncgen_hooks(None, None)
        context.unconfigure()
        glfw.destroy_window(window)

        # allow proper cleanup (workaround for glfw bug)
        end_time = time.perf_counter() + 0.1
        while time.perf_counter() < end_time:
            glfw.wait_events_timeout(end_time - time.perf_counter())


if __name__ == "__main__":
    loop = Loop()
    loop.run()
