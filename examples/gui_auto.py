"""
Run triangle/cube example in an automatically selected GUI backend.
"""

# test_example = true

from wgpu.gui.auto import WgpuCanvas, run

try:
    from .triangle import setup_drawing_sync
except ImportError:
    from triangle import setup_drawing_sync

canvas = WgpuCanvas(title=f"Triangle example on {WgpuCanvas.__name__}")
draw_frame = setup_drawing_sync(canvas)


@canvas.request_draw
def animate():
    draw_frame()
    canvas.request_draw()


if __name__ == "__main__":
    run()
