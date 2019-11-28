import sys
import asyncio

import wgpu
from PyQt5 import QtCore, QtGui, QtWidgets
import qasync

from . import _renderer


class Figure:
    """ Represents the root rectangular region to draw to.
    """

    def __init__(self, title="Figure", size=(640, 480), backend="qt", renderer=None):
        self._views = []
        self._widget = None

        # Select backend
        if backend.lower() == "qt":
            self._widget = QtFigureWidget(None, self)
        else:
            raise ValueError(f"Invalid Figure backend {backend}")

        # Initialize widget, if we have one
        if self._widget is not None:
            self._widget._visvis_set_size(*size)
            self._widget._visvis_set_title(title)

        # Check renderer
        if renderer is None:
            self._renderer = _renderer.SurfaceWgpuRenderer()

    @property
    def views(self):
        return self._views

    @property
    def widget(self):
        return self._widget

    @property
    def size(self):
        return self._widget._visvis_get_size()

    def get_surface_id(self, ctx):
        return self._widget._visvis_get_surface_id(ctx)

    def _on_paint(self):
        if not hasattr(self, "_collected"):
            self._collected = True
            self._renderer.collect_from_figure(self)

        self._renderer.draw_frame(self)


# class QtFigureWidget(QtGui.QWindow):
class QtFigureWidget(QtWidgets.QWidget):
    def __init__(self, parent, figure):
        super().__init__(parent)
        self._figure = figure
        self._surface_id = None

        self.setAttribute(QtCore.Qt.WA_PaintOnScreen, True)
        self.setAutoFillBackground(False)

        self.show()

    def _visvis_set_size(self, width, height):
        self.resize(width, height)

    def _visvis_set_title(self, title):
        self.setWindowTitle(title)

    def _visvis_get_surface_id(self, ctx):  # called by renderer

        # Memoize
        if self._surface_id is not None:
            return self._surface_id

        if sys.platform.startswith("win"):
            # Use create_surface_from_windows_hwnd
            hwnd = wgpu.wgpu_ffi.ffi.cast("void *", int(self.winId()))
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

    def _visvis_get_size(self):
        return self.width(), self.height()

    def paintEvent(self, event):
        self._figure._on_paint()

    def paintEngine(self):
        # https://doc.qt.io/qt-5/qt.html#WidgetAttribute-enum  WA_PaintOnScreen
        return None


app = QtWidgets.QApplication([])
loop = qasync.QEventLoop(app)
asyncio.set_event_loop(loop)

if hasattr(asyncio, "integrate_with_ide"):
    asyncio.integrate_with_ide(loop, run=False)
