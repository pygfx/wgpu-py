"""
Import the viz from triangle.py and run it in a Qt window.
Works with either PySide6, PyQt6, PyQt5 or PySide2.
"""

# For the sake of making this example Just Work, we try multiple QT libs
try:
    from PySide6 import QtWidgets
except ModuleNotFoundError:
    try:
        from PyQt6 import QtWidgets
    except ModuleNotFoundError:
        try:
            from PySide2 import QtWidgets
        except ModuleNotFoundError:
            from PyQt5 import QtWidgets

from wgpu.gui.qt import WgpuCanvas  # WgpuCanvas is a QWidget subclass
import wgpu.backends.rs  # noqa: F401, Select Rust backend

from triangle import main  # The function to call to run the visualization


app = QtWidgets.QApplication([])
canvas = WgpuCanvas(title="wgpu triangle with Qt")

device = main(canvas)

# Enter Qt event loop (compatible with qt5/qt6)
app.exec() if hasattr(app, "exec") else app.exec_()


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
