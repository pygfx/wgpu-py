"""
Import the viz from triangle.py and run it using the auto-gui.
"""
# test_example = true


from wgpu.gui.auto import WgpuCanvas, run
import wgpu.backends.rs  # noqa: F401, Select Rust backend

from triangle import main  # The function to call to run the visualization


canvas = WgpuCanvas()
device = main(canvas)


if __name__ == "__main__":
    run()
