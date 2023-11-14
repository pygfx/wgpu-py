"""
Import the viz from triangle.py and run it in a Qt window.
Works with either PySide6, PyQt6, PyQt5 or PySide2.

# run_example = false
"""
import importlib

# For the sake of making this example Just Work, we try multiple QT libs
for lib in ("PySide6", "PyQt6", "PySide2", "PyQt5"):
    try:
        QtWidgets = importlib.import_module(".QtWidgets", lib)
        break
    except ModuleNotFoundError:
        pass


from wgpu.gui.qt import WgpuCanvas  # WgpuCanvas is a QWidget subclass

from triangle import main  # The function to call to run the visualization


app = QtWidgets.QApplication([])
canvas = WgpuCanvas()

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
