"""
Run triangle/cube example in the glfw GUI backend.
"""

# run_example = false

from wgpu.gui.glfw import WgpuCanvas, run

from triangle import setup_triangle  # noqa: F401, RUF100
from cube import setup_cube  # noqa: F401, RUF100


canvas = WgpuCanvas(title=f"Triangle example on {WgpuCanvas.__name__}")
setup_triangle(canvas)


if __name__ == "__main__":
    run()
