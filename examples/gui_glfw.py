"""
Run triangle/cube example in the glfw GUI backend.
"""

# run_example = false

from wgpu.gui.glfw import WgpuCanvas, run

from triangle import setup_drawing_sync
# from cube import setup_drawing_sync


canvas = WgpuCanvas(title=f"Triangle example on {WgpuCanvas.__name__}")
draw_frame = setup_drawing_sync(canvas)


@canvas.request_draw
def animate():
    draw_frame()
    canvas.request_draw()


if __name__ == "__main__":
    run()
