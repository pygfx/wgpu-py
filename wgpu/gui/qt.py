"""
Support for rendering in a Qt widget. Provides a widget subclass that
can be used as a standalone window or in a larger GUI.
"""

import sys
import ctypes
import importlib

from .base import WgpuCanvasBase

# Select GUI toolkit
for libname in ("PySide2", "PyQt5", "PySide", "PyQt4"):
    if libname in sys.modules:
        QtCore = importlib.import_module(libname + ".QtCore")
        widgets_modname = "QtGui" if QtCore.qVersion()[0] == "4" else "QtWidgets"
        QtWidgets = importlib.import_module(libname + "." + widgets_modname)
        break
else:
    raise ImportError(
        "Import one of PySide2, PyQt5 before the WgpuCanvas to select a Qt toolkit"
    )


def enable_hidpi():
    """ Enable high-res displays.
    """
    try:
        # See https://github.com/pyzo/pyzo/pull/700 why we seem to need both
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # global dpi aware
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor dpi aware
    except Exception:
        pass  # fail on non-windows
    try:
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    except Exception:
        pass  # fail on older Qt's


# If you import this module, you want to use wgpu in a way that does not suck
# on high-res monitors. So we apply the minimal configuration to make this so.
# Most apps probably should also set AA_UseHighDpiPixmaps, but it's not
# needed for wgpu, so not our responsibility (some users may NOT want it set).
enable_hidpi()


class QtWgpuCanvas(WgpuCanvasBase, QtWidgets.QWidget):
    def __init__(self, *args, size=None, title=None, **kwargs):
        super().__init__(*args, **kwargs)

        if size:
            width, height = size
            self.resize(width, height)
        elif "parent" not in kwargs:
            self.resize(640, 480)
        if title:
            self.setWindowTitle(title)

        # The actual surface is held by a subwidget. This is to make sure that
        # the logical surface size is actually integer. Otherwise the window
        # size can be set to subpixel (logical) values, without being able to
        # detect this. See https://github.com/almarklein/wgpu-py/pull/68
        self._subwidget = WgpuSubWidget(self)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        layout.addWidget(self._subwidget)

        self.show()

    # Qt methods

    def update(self):
        super().update()
        self._subwidget.update()

    # Methods that we add from wgpu (snake_case)

    def get_display_id(self):
        # There is qx11info, but it is rarely available.
        # https://doc.qt.io/qt-5/qx11info.html#display
        return super().get_display_id()  # uses X11 lib

    def get_window_id(self):
        return int(self._subwidget.winId())

    def get_pixel_ratio(self):
        # The pixel ratio always seems to be a whole number. When setting
        # the scale in Windows 10 to 175%, Qt pretends its 2.0.
        return self._subwidget.devicePixelRatioF()

    def get_logical_size(self):
        # Sizes in Qt are logical
        lsize = self._subwidget.width(), self._subwidget.height()
        return float(lsize[0]), float(lsize[1])

    def get_physical_size(self):
        # https://doc.qt.io/qt-5/qpaintdevice.html
        # https://doc.qt.io/qt-5/highdpi.html
        lsize = self._subwidget.width(), self._subwidget.height()
        lsize = float(lsize[0]), float(lsize[1])
        ratio = self._subwidget.devicePixelRatioF()
        return round(lsize[0] * ratio), round(lsize[1] * ratio)

    def set_logical_size(self, width, height):
        if width < 0 or height < 0:
            raise ValueError("Window width and height must not be negative")
        self.resize(width, height)  # See note on pixel ratio below

    def request_draw(self):
        self.update()

    def close(self):
        super().close()

    def is_closed(self):
        return not self.isVisible()


class WgpuSubWidget(QtWidgets.QWidget):
    """ The widget that actually provides the surface to render to.
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_PaintOnScreen, True)
        self.setAutoFillBackground(False)

    def paintEngine(self):  # noqa: N802 - this is a Qt method
        # https://doc.qt.io/qt-5/qt.html#WidgetAttribute-enum  WA_PaintOnScreen
        return None

    def paintEvent(self, event):  # noqa: N802 - this is a Qt method
        self.parent()._draw_frame_and_present()


WgpuCanvas = QtWgpuCanvas
