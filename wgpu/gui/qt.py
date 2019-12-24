"""
Support for rendering in a Qt widget. Provides a widget subclass that
can be used as a standalone window or in a larger GUI.
"""

import importlib
import sys

from .base import BaseCanvas


# Select Qt toolkit
qt_libs = ("PySide2", "PyQt5")
for libname in qt_libs:
    # if both are installed, and the user has imported the one that
    # they want to use already, stick with that
    if libname in sys.modules:
        QWidget = sys.modules[libname].QtWidgets.QWidget
        break
else:
    # otherwise, if none have been imported yet, try to do it ourselves
    for libname in qt_libs:
        try:
            QWidget = importlib.import_module(libname + ".QtWidgets").QWidget
            break
        except ImportError:
            pass
    else:
        raise ImportError("Could not import " + ", ".join(qt_libs))


class WgpuCanvas(BaseCanvas, QWidget):
    """ A QWidget subclass that can be used as a canvas to render to.
    """

    def __init__(self, *args, size=(640, 480), title="", **kwargs):
        super().__init__(*args, **kwargs)
        WA_PaintOnScreen = 8  # QtCore.Qt.WA_PaintOnScreen
        self.setAttribute(WA_PaintOnScreen, True)
        self.setAutoFillBackground(False)

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
