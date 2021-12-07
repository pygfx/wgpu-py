"""
Import the viz from triangle.py and run it in a glfw window.
The glfw library can be installed using ``pip install glfw``.
"""

import math
from time import perf_counter

import glfw
from wgpu.gui.glfw import update_glfw_canvasses, WgpuCanvas
import wgpu.backends.rs  # noqa: F401, Select Rust backend

# Import the function that we must call to run the visualization
from triangle import main


glfw.ERROR_REPORTING = "warn"
canvas = WgpuCanvas(title="wgpu triangle with GLFW")
main(canvas)


def simple_event_loop():
    """A real simple event loop, but it keeps the CPU busy."""
    while update_glfw_canvasses():
        glfw.poll_events()


def better_event_loop(max_fps=100):
    """A simple event loop that schedules draws."""
    td = 1 / max_fps
    while update_glfw_canvasses():
        # Determine next time to draw
        now = perf_counter()
        tnext = math.ceil(now / td) * td
        # Process events until it's time to draw
        while now < tnext:
            glfw.wait_events_timeout(tnext - now)
            now = perf_counter()


better_event_loop()
glfw.terminate()
