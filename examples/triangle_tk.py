"""
Import the viz from triangle.py and run it in a Tk window.
Tkinter comes with Python by default.
"""

import tkinter
from wgpu.gui.tk import WgpuCanvas  # WgpuCanvas wraps a glfw window
import wgpu.backend.rs  # noqa: F401, Select Rust backend

# Import the (async) function that we must call to run the visualization
from triangle import main


root = tkinter.Tk()
root.withdraw()
canvas = WgpuCanvas(size=(640, 480), title="wgpu triangle with Tkinter")

main(canvas)

while not canvas.isClosed():
    root.update_idletasks()
    root.update()

# This should work, but the loop does not exit when the window is closed:
# canvas.mainloop()
