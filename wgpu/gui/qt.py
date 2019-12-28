"""
Support for rendering in a Qt widget. Provides a widget subclass that
can be used as a standalone window or in a larger GUI.
"""

import sys

from .base import BaseCanvas


# Select GUI toolkit
for libname in ("PySide2", "PyQt5"):
    if libname in sys.modules:
        QWidget = sys.modules[libname].QtWidgets.QWidget
        break
else:
    raise ImportError("Need to import PySide2.QtWidgets or PyQt5.QtWidgets first.")


class WgpuCanvas(BaseCanvas, QWidget):
    """ A QWidget subclass that can be used as a canvas to render to.
    """

    def __init__(self, *args, size=None, title=None, **kwargs):
        super().__init__(*args, **kwargs)
        WA_PaintOnScreen = 8  # QtCore.Qt.WA_PaintOnScreen
        self.setAttribute(WA_PaintOnScreen, True)
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

    def getWindowId(self):
        return self.winId()
