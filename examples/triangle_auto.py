"""
Import the viz from triangle.py and run it using the auto-gui.
"""
# test_example = true

import sys
from pathlib import Path

from wgpu.gui.auto import WgpuCanvas, run
import wgpu.backends.rs  # noqa: F401, Select Rust backend

sys.path.insert(0, Path(__file__).parent.parent)

from triangle import main  # The function to call to run the visualization


canvas = WgpuCanvas()
device = main(canvas)


if __name__ == "__main__":
    run()
