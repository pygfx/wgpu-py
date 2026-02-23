"""
GUI Auto
--------


Run the triangle example in an automatically selected GUI backend.

The rendercanvas automatically selects one of its available
GUI backends. E.g. running this in a notebook will use the
Jupyter backend. If the Qt event loop is running, it will use the
Qt backend. The default backend uses glfw.

See https://rendercanvas.readthedocs.io/stable/backends.html for more info,
and https://rendercanvas.readthedocs.io/stable/gallery/index.html for examples
with various GUI's and event loops.

"""

# test_example = true

from rendercanvas.auto import RenderCanvas, loop

try:
    from .triangle import setup_drawing_sync
except ImportError:
    from triangle import setup_drawing_sync

canvas = RenderCanvas(title="Cube example on $backend")
draw_frame = setup_drawing_sync(canvas.get_wgpu_context())


@canvas.request_draw
def animate():
    draw_frame()
    canvas.request_draw()


if __name__ == "__main__":
    loop.run()
