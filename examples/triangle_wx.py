"""
Import the viz from triangle.py and run it in a wxPython window.
"""
# run_example = false

import wx
from wgpu.gui.wx import WgpuCanvas

from examples.triangle import main  # The function to call to run the visualization


app = wx.App()
canvas = WgpuCanvas()

main(canvas)
app.MainLoop()
