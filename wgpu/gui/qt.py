"""
Support for rendering in a Qt widget. Provides a widget subclass that
can be used as a standalone window or in a larger GUI.
"""

import ctypes
import importlib
import sys
import traceback

from .base import WgpuCanvasBase, WgpuAutoGui

# Select GUI toolkit
for libname in ("PySide6", "PyQt6", "PySide2", "PyQt5"):
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
        "Before importing wgpu.gui.qt, import one of PySide6/PySide2/PyQt6/PyQt5 to select a Qt toolkit."
    )


BUTTON_MAP = {
    QtCore.Qt.MouseButton.LeftButton: 1,  # == MOUSE_BUTTON_LEFT
    QtCore.Qt.MouseButton.RightButton: 2,  # == MOUSE_BUTTON_RIGHT
    QtCore.Qt.MouseButton.MiddleButton: 3,  # == MOUSE_BUTTON_MIDDLE
    QtCore.Qt.MouseButton.BackButton: 4,
    QtCore.Qt.MouseButton.ForwardButton: 5,
    QtCore.Qt.MouseButton.TaskButton: 6,
    QtCore.Qt.MouseButton.ExtraButton4: 7,
    QtCore.Qt.MouseButton.ExtraButton5: 8,
}

MODIFIERS_MAP = {
    QtCore.Qt.ShiftModifier: "Shift",
    QtCore.Qt.ControlModifier: "Control",
    QtCore.Qt.AltModifier: "Alt",
    QtCore.Qt.MetaModifier: "Meta",
}

KEY_MAP = {
    int(QtCore.Qt.Key_Down): "ArrowDown",
    int(QtCore.Qt.Key_Up): "ArrowUp",
    int(QtCore.Qt.Key_Left): "ArrowLeft",
    int(QtCore.Qt.Key_Right): "ArrowRight",
    int(QtCore.Qt.Key_Backspace): "Backspace",
    int(QtCore.Qt.Key_CapsLock): "CapsLock",
    int(QtCore.Qt.Key_Delete): "Delete",
    int(QtCore.Qt.Key_End): "End",
    int(QtCore.Qt.Key_Enter): "Enter",
    int(QtCore.Qt.Key_Escape): "Escape",
    int(QtCore.Qt.Key_F1): "F1",
    int(QtCore.Qt.Key_F2): "F2",
    int(QtCore.Qt.Key_F3): "F3",
    int(QtCore.Qt.Key_F4): "F4",
    int(QtCore.Qt.Key_F5): "F5",
    int(QtCore.Qt.Key_F6): "F6",
    int(QtCore.Qt.Key_F7): "F7",
    int(QtCore.Qt.Key_F8): "F8",
    int(QtCore.Qt.Key_F9): "F9",
    int(QtCore.Qt.Key_F10): "F10",
    int(QtCore.Qt.Key_F11): "F11",
    int(QtCore.Qt.Key_F12): "F12",
    int(QtCore.Qt.Key_Home): "Home",
    int(QtCore.Qt.Key_Insert): "Insert",
    int(QtCore.Qt.Key_Alt): "Alt",
    int(QtCore.Qt.Key_Control): "Control",
    int(QtCore.Qt.Key_Shift): "Shift",
    int(
        QtCore.Qt.Key_Meta
    ): "Meta",  # meta maps to control in QT on macOS, and vice-versa
    int(QtCore.Qt.Key_NumLock): "NumLock",
    int(QtCore.Qt.Key_PageDown): "PageDown",
    int(QtCore.Qt.Key_PageUp): "Pageup",
    int(QtCore.Qt.Key_Pause): "Pause",
    int(QtCore.Qt.Key_ScrollLock): "ScrollLock",
    int(QtCore.Qt.Key_Tab): "Tab",
}


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


class QWgpuWidget(WgpuCanvasBase, QtWidgets.QWidget):
    """A QWidget representing a wgpu canvas that can be embedded in a Qt application."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Configure how Qt renders this widget
        self.setAttribute(WA_PaintOnScreen, True)
        self.setAutoFillBackground(False)

        # A timer for limiting fps
        self._request_draw_timer = QtCore.QTimer()
        self._request_draw_timer.setTimerType(PreciseTimer)
        self._request_draw_timer.setSingleShot(True)
        self._request_draw_timer.timeout.connect(self.update)

        # Get the window id one time. For some reason this is needed
        # to "activate" the canvas. Otherwise the viz is not shown if
        # one does not provide canvas to request_adapter().
        self.get_window_id()

    def paintEngine(self):  # noqa: N802 - this is a Qt method
        # https://doc.qt.io/qt-5/qt.html#WidgetAttribute-enum  WA_PaintOnScreen
        return None

    def paintEvent(self, event):  # noqa: N802 - this is a Qt method
        self._draw_frame_and_present()

    # Methods that we add from wgpu (snake_case)

    def get_display_id(self):
        # There is qx11info, but it is rarely available.
        # https://doc.qt.io/qt-5/qx11info.html#display
        return super().get_display_id()  # uses X11 lib

    def get_window_id(self):
        return int(self.winId())

    def get_pixel_ratio(self):
        # Observations:
        # * On Win10 + PyQt5 the ratio is a whole number (175% becomes 2).
        # * On Win10 + PyQt6 the ratio is correct (non-integer).
        return self.devicePixelRatioF()

    def get_logical_size(self):
        # Sizes in Qt are logical
        lsize = self.width(), self.height()
        return float(lsize[0]), float(lsize[1])

    def get_physical_size(self):
        # https://doc.qt.io/qt-5/qpaintdevice.html
        # https://doc.qt.io/qt-5/highdpi.html
        lsize = self.width(), self.height()
        lsize = float(lsize[0]), float(lsize[1])
        ratio = self.devicePixelRatioF()
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
        self.resize(width, height)  # See comment on pixel ratio

    def _request_draw(self):
        if not self._request_draw_timer.isActive():
            self._request_draw_timer.start(self._get_draw_wait_time() * 1000)

    def close(self):
        super().close()

    def is_closed(self):
        return not self.isVisible()


class QWgpuCanvas(WgpuAutoGui, WgpuCanvasBase, QtWidgets.QWidget):
    """A toplevel Qt widget providing a wgpu canvas."""

    # Most of this is proxying stuff to the inner widget.
    # We cannot use a toplevel widget directly, otherwise the window
    # size can be set to subpixel (logical) values, without being able to
    # detect this. See https://github.com/pygfx/wgpu-py/pull/68

    def __init__(self, *, size=None, title=None, max_fps=30, **kwargs):
        # When using Qt, there needs to be an
        # application before any widget is created
        get_app()

        super().__init__(**kwargs)

        self.set_logical_size(*(size or (640, 480)))
        self.setWindowTitle(title or "qt wgpu canvas")

        self._subwidget = QWgpuWidget(self, max_fps=max_fps)

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
        return self._subwidget.get_display_id()

    def get_window_id(self):
        return self._subwidget.get_window_id()

    def get_pixel_ratio(self):
        return self._subwidget.get_pixel_ratio()

    def get_logical_size(self):
        return self._subwidget.get_logical_size()

    def get_physical_size(self):
        return self._subwidget.get_physical_size()

    def set_logical_size(self, width, height):
        if width < 0 or height < 0:
            raise ValueError("Window width and height must not be negative")
        self.resize(width, height)  # See comment on pixel ratio

    def _request_draw(self):
        return self._subwidget._request_draw()

    def close(self):
        super().close()

    def is_closed(self):
        return not self.isVisible()

    # Methods that we need to explicitly delegate to the subwidget

    def get_context(self, *args, **kwargs):
        return self._subwidget.get_context(*args, **kwargs)

    def request_draw(self, *args, **kwargs):
        return self._subwidget.request_draw(*args, **kwargs)

    # Auto event API

    def _emit_event(self, event):
        try:
            self.handle_event(event)
        except Exception:
            # Print exception and store exc info for postmortem debugging
            exc_info = list(sys.exc_info())
            exc_info[2] = exc_info[2].tb_next  # skip *this* function
            sys.last_type, sys.last_value, sys.last_traceback = exc_info
            traceback.print_exception(*exc_info)

    # User events to jupyter_rfb events

    def _key_event(self, event_type, event):
        modifiers = [
            MODIFIERS_MAP[mod]
            for mod in MODIFIERS_MAP.keys()
            if mod & event.modifiers()
        ]

        ev = {
            "event_type": event_type,
            "key": KEY_MAP.get(event.key(), event.text()),
            "modifiers": modifiers,
        }
        self._emit_event(ev)

    def keyPressEvent(self, event):  # noqa: N802
        self._key_event("key_down", event)

    def keyReleaseEvent(self, event):  # noqa: N802
        self._key_event("key_up", event)

    def _mouse_event(self, event_type, event, touches=True):
        button = BUTTON_MAP.get(event.button(), 0)
        buttons = [
            BUTTON_MAP[button]
            for button in BUTTON_MAP.keys()
            if button & event.buttons()
        ]

        # For Qt on macOS Control and Meta are switched
        modifiers = [
            MODIFIERS_MAP[mod]
            for mod in MODIFIERS_MAP.keys()
            if mod & event.modifiers()
        ]

        ev = {
            "event_type": event_type,
            "x": event.x(),
            "y": event.y(),
            "button": button,
            "buttons": buttons,
            "modifiers": modifiers,
        }
        if touches:
            ev.update(
                {
                    "ntouches": 0,  # TODO
                    "touches": {},  # TODO
                }
            )
        self._emit_event(ev)

    def mousePressEvent(self, event):  # noqa: N802
        self._mouse_event("pointer_down", event)

    def mouseMoveEvent(self, event):  # noqa: N802
        self._mouse_event("pointer_move", event)

    def mouseReleaseEvent(self, event):  # noqa: N802
        self._mouse_event("pointer_up", event)

    def mouseDoubleClickEvent(self, event):  # noqa: N802
        self._mouse_event("double_click", event, touches=False)

    def wheelEvent(self, event):  # noqa: N802
        # For Qt on macOS Control and Meta are switched
        modifiers = [
            MODIFIERS_MAP[mod]
            for mod in MODIFIERS_MAP.keys()
            if mod & event.modifiers()
        ]

        ev = {
            "event_type": "wheel",
            "dx": -event.angleDelta().x(),
            "dy": -event.angleDelta().y(),
            "x": event.position().x(),
            "y": event.position().y(),
            "modifiers": modifiers,
        }
        self._emit_event(ev)

    def resizeEvent(self, event):  # noqa: N802
        ev = {
            "event_type": "resize",
            "width": float(event.size().width()),
            "height": float(event.size().height()),
            "pixel_ratio": self.get_pixel_ratio(),
        }
        self._emit_event(ev)

    def closeEvent(self, event):  # noqa: N802
        self._emit_event({"event_type": "close"})


# Make available under a name that is the same for all gui backends
WgpuWidget = QWgpuWidget
WgpuCanvas = QWgpuCanvas


def get_app():
    """Return global instance of Qt app instance or create one if not created yet."""
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def run():
    app = get_app()
    app.exec() if hasattr(app, "exec") else app.exec_()


def call_later(delay, callback, *args):
    QtCore.QTimer.singleShot(delay * 1000, lambda: callback(*args))
