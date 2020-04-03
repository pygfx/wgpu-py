"""
Import the viz from triangle.py and run it in a glfw window,
while we "integrate" glfw with an asyncio event loop.
The glfw library can be installed using ``pip install glfw``.
"""

import asyncio  # noqa: E402

import glfw
from wgpu.gui.glfw import update_glfw_canvasses, WgpuCanvas
import wgpu.backends.rs  # noqa: F401, Select Rust backend

# Import the (async) function that we must call to run the visualization
from triangle import main_async


glfw.init()
canvas = WgpuCanvas(size=(640, 480), title="wgpu triangle with GLFW")


async def mainloop():
    await main_async(canvas)
    while update_glfw_canvasses():
        await asyncio.sleep(0.001)
        glfw.poll_events()
    loop.stop()
    glfw.terminate()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(mainloop())
    loop.run_forever()
