import sys
import asyncio

import wgpu
from PyQt5 import QtCore, QtGui, QtWidgets
import qasync


class Figure:
    """ Represents a widget. Either a standalone window or embedded in an application.
    Qt only for now.
    """

    def __init__(self):

        self._widget = None
        self._views = []
        self._surface_id = None

    def _get_surface_id(self, ctx):
        raise NotImplementedError()

    def get_size(self):
        raise NotImplementedError()

    @property
    def widget(self):
        return self._widget

    @property
    def views(self):
        return self._views


class QtFigure(Figure):
    def __init__(self, width=640, height=480, name="visvis2 or something"):
        super().__init__()

        self._widget = QtGui.QWindow(None)

        # Use winId() or effectiveWinId() to get the Windows window handle

        self._widget.resize(width, height)
        self._widget.setTitle(name)
        self._widget.show()

        # async def _keep_qt_alive():
        #     while self._widget.isVisible():  # todo: while any windows alive ... and move to module level
        #         await asyncio.sleep(0.1)
        #         app.flush()
        #         app.processEvents()
        #
        # asyncio.get_event_loop().create_task(_keep_qt_alive())

    def _get_surface_id(self, ctx):  # called by renderer

        # Memoize
        if self._surface_id is not None:
            return self._surface_id

        if sys.platform.startswith("win"):
            # Use create_surface_from_windows_hwnd
            hwnd = wgpu.wgpu_ffi.ffi.cast("void *", int(self._widget.winId()))
            hinstance = wgpu.wgpu_ffi.ffi.NULL
            surface_id = ctx.create_surface_from_windows_hwnd(hinstance, hwnd)

        elif sys.platform.startswith("linux"):
            # Use create_surface_from_xlib
            raise NotImplementedError("Linux")

        elif sys.platform.startswith("darwin"):
            # Use create_surface_from_metal_layer
            raise NotImplementedError("OS-X")

        else:
            raise RuntimeError("Unsupported platform")

        self._surface_id = surface_id
        return surface_id

    def get_size(self):
        return self._widget.width(), self._widget.height()


app = QtWidgets.QApplication([])
loop = qasync.QEventLoop(app)
asyncio.set_event_loop(loop)
