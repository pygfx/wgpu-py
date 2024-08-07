"""
Import the viz from triangle.py and run it in a wxPython window.
"""

# run_example = false

import wx
from wgpu.gui.wx import WgpuCanvas

from triangle import main  # The function to call to run the visualization


class MyCanvas(WgpuCanvas):
    def handle_event(self, event):
        if event["event_type"] != "pointer_move":
            print(event)


app = wx.App()
canvas = MyCanvas()

main(canvas)
app.MainLoop()
