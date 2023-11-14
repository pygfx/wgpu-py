"""
Import the viz from triangle.py and run it using glfw (which uses asyncio for the event loop).

# run_example = false
"""

import sys
from pathlib import Path

from wgpu.gui.glfw import WgpuCanvas, run

sys.path.insert(0, str(Path(__file__).parent))

from triangle import main  # noqa: E402, The function to call to run the visualization


canvas = WgpuCanvas()
device = main(canvas)


if __name__ == "__main__":
    run()
