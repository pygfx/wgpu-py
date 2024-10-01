"""
Run the triangle/cube example in as custom event loop based on asyncio.
It uses the asynchronous path, which calls the async versions of the wgpu API.

Uses glfw as a GUI because its loop-agnostic.
"""

# run_example = false

import time
import asyncio

import glfw

from gui_direct import MinimalGlfwCanvas
from triangle import setup_triangle_async  # noqa: F401, RUF100
from cube import setup_cube_async  # noqa: F401, RUF100


async def main_loop():
    # create a window with glfw
    glfw.init()
    # disable automatic API selection, we are not using opengl
    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
    glfw.window_hint(glfw.RESIZABLE, True)
    window = glfw.create_window(640, 480, "wgpu with asyncio", None, None)

    # create canvas
    canvas = MinimalGlfwCanvas(window)
    # await setup_triangle_async(canvas)
    await setup_cube_async(canvas)

    last_frame_time = time.perf_counter()
    frame_count = 0

    while True:
        await asyncio.sleep(0.01)
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
    glfw.hide_window(window)
    glfw.destroy_window(window)
    glfw.terminate()


if __name__ == "__main__":
    asyncio.run(main_loop())
