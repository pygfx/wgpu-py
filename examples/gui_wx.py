"""
Run the triangle/cube example in the wx GUI backend.
"""

# run_example = false

import wx
from wgpu.gui.wx import WgpuCanvas

from triangle import setup_drawing_sync
# from cube import setup_drawing_sync


app = wx.App()
canvas = WgpuCanvas(title=f"Triangle example on {WgpuCanvas.__name__}")

draw_func = setup_drawing_sync(canvas)
canvas.request_draw(draw_func)

app.MainLoop()
