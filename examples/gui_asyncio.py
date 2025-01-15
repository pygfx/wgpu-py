"""
Run the triangle/cube example in as custom event loop based on asyncio.
It uses the asynchronous path, which calls the async versions of the wgpu API.

Uses glfw as a GUI because it's loop-agnostic.
"""

# run_example = false

import time
import asyncio

import glfw

from gui_direct import MinimalGlfwCanvas, poll_glfw_briefly

# from triangle import setup_drawing_async
from cube import setup_drawing_async


async def main_loop():
    # create canvas
    canvas = MinimalGlfwCanvas("wgpu gui asyncio")
    draw_frame = await setup_drawing_async(canvas)

    last_frame_time = time.perf_counter()
    frame_count = 0

    while not glfw.window_should_close(canvas.window):
        await asyncio.sleep(0.01)
        # process inputs
        glfw.poll_events()
        # draw a frame
        await draw_frame()
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
    poll_glfw_briefly()


if __name__ == "__main__":
    asyncio.run(main_loop())
