"""
Pygfx Example
=============
Placeholder for just trying something simple, right now this is from hellow_triangle.py
"""


import os
os.environ["PYGFX_DISABLE_SYSTEM_FONTS"] = "1"

from rendercanvas.auto import RenderCanvas, loop
import pygfx as gfx


canvas = RenderCanvas()
print(f"{canvas=}")
renderer = gfx.WgpuRenderer(canvas)
print(f"{renderer=}")
camera = gfx.NDCCamera()
print(f"{camera=}")

triangle = gfx.Mesh(
    gfx.Geometry(
        indices=[(0, 1, 2)],
        positions=[(0.0, -0.5, 0), (0.5, 0.5, 0), (-0.5, 0.75, 0)],
        colors=[(1, 1, 0, 1), (1, 0, 1, 1), (0, 1, 1, 1)],
    ),
    gfx.MeshBasicMaterial(color_mode="vertex"),
)
print(triangle)


if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(triangle, camera))
    loop.run()