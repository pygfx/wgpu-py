"""
Run the triangle/cube example in the wx GUI backend.
"""

# run_example = false

import wx
from rendercanvas.wx import RenderCanvas

from triangle import setup_drawing_sync
# from cube import setup_drawing_sync


app = wx.App()
canvas = RenderCanvas(title=f"Triangle example on {RenderCanvas .__name__}")

draw_func = setup_drawing_sync(canvas)
canvas.request_draw(draw_func)

app.MainLoop()
