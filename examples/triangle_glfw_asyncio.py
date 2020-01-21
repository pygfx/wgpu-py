"""
Import the viz from triangle.py and run it in a glfw window,
while we "integrate" glfw with an asyncio event loop.
The glfw library can be installed using ``pip install glfw``.
"""

import asyncio  # noqa: E402

import glfw
from wgpu.gui.glfw import WgpuCanvas  # WgpuCanvas wraps a glfw window
import wgpu.backend.rs  # noqa: F401, Select Rust backend

# Import the (async) function that we must call to run the visualization
from triangle import mainAsync


glfw.init()
canvas = WgpuCanvas(size=(640, 480), title="wgpu triangle with GLFW")


async def mainLoop():
    await mainAsync(canvas)
    while not canvas.isClosed():
        await asyncio.sleep(0.001)
        glfw.poll_events()
    loop.stop()
    glfw.terminate()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(mainLoop())
    loop.run_forever()
