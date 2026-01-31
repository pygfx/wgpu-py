"""
Pygfx Example
=============
Placeholder for just trying something simple, right now this is from hello_triangle.py
"""


# import os
# os.environ["PYGFX_DISABLE_SYSTEM_FONTS"] = "1"

from rendercanvas.auto import RenderCanvas, loop
import pygfx as gfx


canvas = RenderCanvas()
renderer = gfx.WgpuRenderer(canvas, ppaa="none")
camera = gfx.NDCCamera()

triangle = gfx.Mesh(
    gfx.Geometry(
        indices=[(0, 1, 2)],
        positions=[(0.0, -0.5, 0), (0.5, 0.5, 0), (-0.5, 0.75, 0)],
        colors=[(1, 1, 0, 1), (1, 0, 1, 1), (0, 1, 1, 1)],
    ),
    gfx.MeshBasicMaterial(color_mode="vertex"),
)


if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(triangle, camera))
    loop.run()
