"""
Support for rendering in a Qt widget. Provides a widget subclass that
can be used as a standalone window or in a larger GUI.
"""

import sys
import importlib

from .base import BaseCanvas

# Select GUI toolkit
for libname in ("PySide2", "PyQt5", "PySide", "PyQt4"):
    if libname in sys.modules:
        QtCore = importlib.import_module(libname + ".QtCore")
        widgets_modname = "QtGui" if QtCore.qVersion()[0] == "4" else "QtWidgets"
        QWidget = importlib.import_module(libname + "." + widgets_modname).QWidget
        break
else:
    raise ImportError(
        "Import one of PySide2, PyQt5 before the WgpuCanvas to select a Qt toolkit"
    )


class QtWgpuCanvas(BaseCanvas, QWidget):
    """ A QWidget subclass that can be used as a canvas to render to.
    """

    def __init__(self, *args, size=None, title=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAttribute(QtCore.Qt.WA_PaintOnScreen, True)
        self.setAutoFillBackground(False)

        if size:
            width, height = size
            self.resize(width, height)
        if title:
            self.setWindowTitle(title)
        self.show()

    def paintEngine(self):
        # https://doc.qt.io/qt-5/qt.html#WidgetAttribute-enum  WA_PaintOnScreen
        return None

    def paintEvent(self, event):
        self._drawFrameAndPresent()

    def getSizeAndPixelRatio(self):
        pixelratio = 1  # todo: pixelratio
        return self.width(), self.height(), pixelratio

    def isClosed(self):
        return not self.isVisible()

    def getDisplayId(self):
        # There is qx11info, but it is rarely available.
        # https://doc.qt.io/qt-5/qx11info.html#display
        return super().getDisplayId()  # use X11 lib

    def getWindowId(self):
        return int(self.winId())


WgpuCanvas = QtWgpuCanvas
