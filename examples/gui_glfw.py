"""
Run triangle/cube example in the glfw rendercanvas backend.
"""

# run_example = false

from rendercanvas.glfw import GlfwRenderCanvas, loop

# from triangle import setup_drawing_sync
from cube import setup_drawing_sync


canvas = GlfwRenderCanvas(title=f"Triangle example on {GlfwRenderCanvas.__name__}")
draw_frame = setup_drawing_sync(canvas)


@canvas.request_draw
def animate():
    draw_frame()
    canvas.request_draw()


if __name__ == "__main__":
    loop.run()
