"""
An example showing that with WGPU you can draw to the window of another
process. Just a proof of concept, this is far from perfect yet:

* It works if I run it in Pyzo, but not if I run it from the terminal.
* I only tried it on Windows.
* You'll want to let the proxy know about size changes.
* The request_draw should invoke a draw (in asyncio?), not draw directly.
* Properly handling closing the figure (from both ends).
"""

# run_example = false

import sys
import json
import time
import subprocess

from rendercanvas.asyncio import loop
from rendercanvas.base import BaseRenderCanvas, BaseCanvasGroup

# Import the function that we must call to run the visualization
# from triangle import setup_drawing_sync
from cube import setup_drawing_sync


code = """
import sys
import json
from PyQt6 import QtWidgets  # Use either PySide6 or PyQt6
from rendercanvas.qt import RenderCanvas

app = QtWidgets.QApplication([])
canvas = RenderCanvas(title="wgpu cube in Qt subprocess", update_mode="ondemand")

print(json.dumps(canvas._subwidget._rc_get_present_methods()))
print(canvas.get_physical_size())
sys.stdout.flush()

app.exec()
"""


class ProxyCanvasGroup(BaseCanvasGroup):
    pass


class ProxyCanvas(BaseRenderCanvas):
    _rc_canvas_group = ProxyCanvasGroup(loop)

    def __init__(self):
        super().__init__()
        self._present_methods = json.loads(p.stdout.readline().decode())
        print(self._present_methods)
        self._psize = tuple(
            int(x) for x in p.stdout.readline().decode().strip().strip("()").split(",")
        )
        print(self._psize)
        time.sleep(0.2)

    def _rc_get_present_methods(self):
        return self._present_methods

    def _rc_request_draw(self):
        loop = self._rc_canvas_group.get_loop()
        loop.call_soon(self._draw_frame_and_present)

    def _rc_get_physical_size(self):
        return self._psize

    def _rc_get_pixel_ratio(self):
        return 1

    def _rc_get_logical_size(self):
        return self._psize

    def _rc_set_logical_size(self, width, height):
        pass

    def _rc_close(self):
        p.kill()

    def _rc_get_closed(self):
        return_code = p.poll()
        return return_code is not None


# Create subprocess
p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE)

# Create a canvas that maps to the window of that subprocess
canvas = ProxyCanvas()
canvas.set_update_mode("continuous")

# Register our draw function
draw_frame = setup_drawing_sync(canvas)
canvas.request_draw(draw_frame)

# Start the event loop
loop.run()
