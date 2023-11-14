"""
An example showing that with WGPU you can draw to the window of another
process. Just a proof of concept, this is far from perfect yet:

* It works if I run it in Pyzo, but not if I run it from the terminal.
* I only tried it on Windows.
* You'll want to let the proxy know about size changes.
* The request_draw should invoke a draw (in asyncio?), not draw directly.
* Properly handling closing the figure (from both ends).

# run_example = false
"""

import sys
import time
import subprocess

from wgpu.gui import WgpuCanvasBase

# Import the (async) function that we must call to run the visualization
from triangle import main


code = """
import sys
from PySide6 import QtWidgets  # Use either PySide6 or PyQt6
from wgpu.gui.qt import WgpuCanvas

app = QtWidgets.QApplication([])
canvas = WgpuCanvas(title="wgpu triangle in Qt subprocess")

print(canvas.get_window_id())
#print(canvas.get_display_id())
print(canvas.get_physical_size())
sys.stdout.flush()

app.exec_()
"""


class ProxyCanvas(WgpuCanvasBase):
    def __init__(self):
        super().__init__()
        self._window_id = int(p.stdout.readline().decode())
        self._psize = tuple(
            int(x) for x in p.stdout.readline().decode().strip().strip("()").split(",")
        )
        print(self._psize)
        time.sleep(0.2)

    def get_window_id(self):
        return self._window_id

    def get_physical_size(self):
        return self._psize

    def get_pixel_ratio(self):
        return 1

    def get_logical_size(self):
        return self._psize

    def set_logical_size(self, width, height):
        pass

    def close(self):
        p.kill()

    def is_closed(self):
        raise NotImplementedError()

    def _request_draw(self):
        self.draw_frame()


# Create subprocess
p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE)

# Create a canvas that maps to the window of that subprocess
canvas = ProxyCanvas()

# Go!
main(canvas)
time.sleep(3)
