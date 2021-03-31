"""
Support for rendering in a Qt widget. Provides a widget subclass that
can be used as a standalone window or in a larger GUI.
"""

import sys
import time
import ctypes
import importlib

from .base import WgpuCanvasBase

# Select GUI toolkit
for libname in ("PySide6", "PyQt6", "PySide2", "PyQt5", "PySide", "PyQt4"):
    if libname in sys.modules:
        QtCore = importlib.import_module(libname + ".QtCore")
        widgets_modname = "QtGui" if QtCore.qVersion()[0] == "4" else "QtWidgets"
        QtWidgets = importlib.import_module(libname + "." + widgets_modname)
        try:
            WA_PaintOnScreen = QtCore.Qt.WidgetAttribute.WA_PaintOnScreen
            PreciseTimer = QtCore.Qt.TimerType.PreciseTimer
        except AttributeError:
            WA_PaintOnScreen = QtCore.Qt.WA_PaintOnScreen
            PreciseTimer = QtCore.Qt.PreciseTimer
        break
else:
    raise ImportError(
        "Import one of PySide2, PyQt5 before the WgpuCanvas to select a Qt toolkit"
    )

# Make Qt not ignore XDG_SESSION_TYPE
# is_wayland = "wayland" in os.getenv("XDG_SESSION_TYPE", "").lower()
# if is_wayland:
#     os.environ["QT_QPA_PLATFORM"] = "wayland"


def enable_hidpi():
    """Enable high-res displays."""
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
    """A Qt widget providing a wgpu canvas."""

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
        # detect this. See https://github.com/pygfx/wgpu-py/pull/68
        self._subwidget = WgpuSubWidget(self)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        layout.addWidget(self._subwidget)

        # A timer for limiting fps
        self._target_fps = 30  # subclasses could edit this value
        self._request_draw_timer = QtCore.QTimer()
        self._request_draw_timer.setTimerType(PreciseTimer)
        self._request_draw_timer.setSingleShot(True)
        self._request_draw_timer.timeout.connect(self.update)

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
        # Observations:
        # * On Win10 + PyQt5 the ratio is a whole number (175% becomes 2).
        # * On Win10 + PyQt6 the ratio is correct (non-integer).
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
        # When the ratio is not integer (qt6), we need to somehow round
        # it. It turns out that we need to round it, but also add a
        # small offset. Tested on Win10 with several different OS
        # scales. Would be nice if we could ask Qt for the exact
        # physical size! Not an issue on qt5, because ratio is always
        # integer then.
        return round(lsize[0] * ratio + 0.01), round(lsize[1] * ratio + 0.01)

    def set_logical_size(self, width, height):
        if width < 0 or height < 0:
            raise ValueError("Window width and height must not be negative")
        self.resize(width, height)  # See note on pixel ratio below

    def _request_draw(self):
        if not self._request_draw_timer.isActive():
            now = time.perf_counter()
            target_time = self._subwidget._draw_time + 1 / self._target_fps
            wait_time = max(0, target_time - now)
            self._request_draw_timer.start(wait_time * 1000)

    def close(self):
        super().close()

    def is_closed(self):
        return not self.isVisible()


class WgpuSubWidget(QtWidgets.QWidget):
    """The widget that actually provides the surface to render to."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(WA_PaintOnScreen, True)
        self.setAutoFillBackground(False)
        self._draw_time = 0

    def paintEngine(self):  # noqa: N802 - this is a Qt method
        # https://doc.qt.io/qt-5/qt.html#WidgetAttribute-enum  WA_PaintOnScreen
        return None

    def paintEvent(self, event):  # noqa: N802 - this is a Qt method
        self._draw_time = time.perf_counter()
        self.parent()._draw_frame_and_present()


# Make available under a name that is the same for all gui backends
WgpuCanvas = QtWgpuCanvas
