"""
Run the triangle/cube example in the wx GUI backend.
"""

# run_example = false

import wx
from rendercanvas.wx import RenderCanvas

from triangle import setup_drawing_sync
# from cube import setup_drawing_sync


app = wx.App()
canvas = RenderCanvas(title="Triangle example on $backend")

draw_func = setup_drawing_sync(canvas)
canvas.request_draw(draw_func)

app.MainLoop()
