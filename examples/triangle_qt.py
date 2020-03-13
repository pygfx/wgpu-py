"""
Import the viz from triangle.py and run it in a Qt window.
Works with either PyQt5 or PySide2.
"""

from PyQt5 import QtWidgets  # Use either PyQt5 or Pyside2
from wgpu.gui.qt import WgpuCanvas  # WgpuCanvas is a QWidget subclass
import wgpu.backends.rs  # noqa: F401, Select Rust backend

# Import the (async) function that we must call to run the visualization
from triangle import main


app = QtWidgets.QApplication([])
canvas = WgpuCanvas(title="wgpu triangle with Qt")

main(canvas)
app.exec_()


# For those interested, this is a simple way to integrate Qt's event
# loop with asyncio, but for real apps you probably want to use
# something like the qasync library.
# async def mainloop():
#     await main_async(canvas)
#     while not canvas.is_closed():
#         await asyncio.sleep(0.001)
#         app.flush()
#         app.processEvents()
#     loop.stop()
