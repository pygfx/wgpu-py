import sys

from .base import BaseCanvas


# Select GUI toolkit
for libname in ("PySide2", "PyQt5"):
    if libname in sys.modules:
        QWidget = sys.modules[libname].QtWidgets.QWidget
        break
else:
    raise ImportError("Need to import PySide2.QtWidgets or PyQt5.QtWidgets first.")


class QGpuWidget(BaseCanvas, QWidget):
    """ A QWidget subclass that can be used as a canvas to render to.
    Implements methods:

    * getSizeAndPixelRatio()
    * configureSwapChain()
    * setDrawFunction()
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._err_hashes = {}

        WA_PaintOnScreen = 8  # QtCore.Qt.WA_PaintOnScreen
        self.setAttribute(WA_PaintOnScreen, True)
        self.setAutoFillBackground(False)

    def paintEngine(self):
        # https://doc.qt.io/qt-5/qt.html#WidgetAttribute-enum  WA_PaintOnScreen
        return None

    def paintEvent(self, event):
        try:
            if self._drawFunction is not None:
                self._drawFunction()
            if self._swapchain is not None:
                self._swapchain._gui_present()  # noqa - a.k.a swap buffers
        except Exception:
            # Enable PM debuging
            sys.last_type, sys.last_value, sys.last_traceback = sys.exc_info()
            msg = str(sys.last_value)
            msgh = hash(msg)
            if msgh in self._err_hashes:
                count = self._err_hashes[msgh] + 1
                self._err_hashes[msgh] = count
                shortmsg = msg.split("\n", 1)[0].strip()[:50]
                sys.stderr.write(f"Error in draw again ({count}): {shortmsg}\n")
            else:
                self._err_hashes[msgh] = 1
                sys.stderr.write(f"Error in draw: " + msg.strip() + "\n")
                traceback.print_last(6)

    def getSizeAndPixelRatio(self):
        pixelratio = 1  # todo: pixelratio
        return self.width(), self.height(), pixelratio

    def getWindowId(self):
        return self.winId()
