"""
Import the viz from triangle.py and run it in a Tk window.
Tkinter comes with Python by default.
"""

import asyncio

import tkinter
from wgpu.gui.tk import WgpuCanvas  # WgpuCanvas wraps a glfw window
import wgpu.backend.rs  # noqa: F401, Select Rust backend

# Import the (async) function that we must call to run the visualization
from triangle import main


root = tkinter.Tk()
root.withdraw()
canvas = WgpuCanvas(size=(640, 480), title="wgpu triangle with Tkinter")


async def mainLoop():
    await main(canvas)
    while not canvas.isClosed():
        await asyncio.sleep(0.001)
        root.update_idletasks()
        root.update()
    loop.stop()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(mainLoop())
    loop.run_forever()
