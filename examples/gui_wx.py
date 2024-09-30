"""
Run the triangle/cube example in the wx GUI backend.
"""

# run_example = false

import wx
from wgpu.gui.wx import WgpuCanvas

from triangle import setup_triangle  # noqa: F401, RUF100
from cube import setup_cube  # noqa: F401, RUF100


app = wx.App()
canvas = WgpuCanvas(title=f"Triangle example on {WgpuCanvas.__name__}")

setup_triangle(canvas)
app.MainLoop()
