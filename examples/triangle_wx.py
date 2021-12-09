"""
Import the viz from triangle.py and run it in a wxPython window.
"""

import wx
from wgpu.gui.wx import WgpuCanvas
import wgpu.backends.rs  # noqa: F401, Select Rust backend

from examples.triangle import main  # The function to call to run the visualization


app = wx.App()
canvas = WgpuCanvas()

main(canvas)
app.MainLoop()
