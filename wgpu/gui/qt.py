"""
Support for rendering in a Qt widget. Provides a widget subclass that
can be used as a standalone window or in a larger GUI.
"""

from importlib import import_module
import sys

from .base import BaseCanvas


# Select Qt toolkit
libs = ("PySide2", "PyQt5")
QWidget = None
for libname in libs:
    # if one of the libs is already imported, use that
    if libname in sys.modules:
        QWidget = sys.modules[libname].QtWidgets.QWidget
        break
else:
    # otherwise, see which libs are available
    available = {}
    for libname in libs:
        try:
            available[libname] = import_module(libname + ".QtWidgets").QWidget
        except ImportError:
            pass
    if len(available) == 1:
        # if only one is available, use that
        QWidget = list(available.values())[0]
    else:
        # otherwise error out and force the user to make a choice
        raise ImportError("Import one of " + ", ".join(available) + " before "
                          "the WgpuCanvas to select a Qt toolkit")
if QWidget is None:
    raise ImportError("Could not import " + " or ".join(libs))


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
