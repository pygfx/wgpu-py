"""
Support for rendering in a Qt widget. Provides a widget subclass that
can be used as a standalone window or in a larger GUI.
"""

import sys
import ctypes
import importlib

from .base import WgpuCanvasBase, WgpuLoop, WgpuTimer, pop_kwargs_for_base_canvas
from ._gui_utils import (
    logger,
    SYSTEM_IS_WAYLAND,
    get_alt_x11_display,
    get_alt_wayland_display,
    get_imported_qt_lib,
)


# Select GUI toolkit
libname, already_had_app_on_import = get_imported_qt_lib()
if libname:
    QtCore = importlib.import_module(".QtCore", libname)
    QtGui = importlib.import_module(".QtGui", libname)
    QtWidgets = importlib.import_module(".QtWidgets", libname)
    try:
        WA_PaintOnScreen = QtCore.Qt.WidgetAttribute.WA_PaintOnScreen
        WA_DeleteOnClose = QtCore.Qt.WidgetAttribute.WA_DeleteOnClose
        WA_InputMethodEnabled = QtCore.Qt.WidgetAttribute.WA_InputMethodEnabled
        PreciseTimer = QtCore.Qt.TimerType.PreciseTimer
        KeyboardModifiers = QtCore.Qt.KeyboardModifier
        FocusPolicy = QtCore.Qt.FocusPolicy
        Keys = QtCore.Qt.Key
    except AttributeError:
        WA_PaintOnScreen = QtCore.Qt.WA_PaintOnScreen
        WA_DeleteOnClose = QtCore.Qt.WA_DeleteOnClose
        WA_InputMethodEnabled = QtCore.Qt.WA_InputMethodEnabled
        PreciseTimer = QtCore.Qt.PreciseTimer
        KeyboardModifiers = QtCore.Qt
        FocusPolicy = QtCore.Qt
        Keys = QtCore.Qt
else:
    raise ImportError(
        "Before importing wgpu.gui.qt, import one of PySide6/PySide2/PyQt6/PyQt5 to select a Qt toolkit."
    )


# Get version
if libname.startswith("PySide"):
    qt_version_info = QtCore.__version_info__
else:
    try:
        qt_version_info = tuple(int(i) for i in QtCore.QT_VERSION_STR.split(".")[:3])
    except Exception:  # Failsafe
        qt_version_info = (0, 0, 0)


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
    KeyboardModifiers.ShiftModifier: "Shift",
    KeyboardModifiers.ControlModifier: "Control",
    KeyboardModifiers.AltModifier: "Alt",
    KeyboardModifiers.MetaModifier: "Meta",
}

KEY_MAP = {
    int(Keys.Key_Down): "ArrowDown",
    int(Keys.Key_Up): "ArrowUp",
    int(Keys.Key_Left): "ArrowLeft",
    int(Keys.Key_Right): "ArrowRight",
    int(Keys.Key_Backspace): "Backspace",
    int(Keys.Key_CapsLock): "CapsLock",
    int(Keys.Key_Delete): "Delete",
    int(Keys.Key_End): "End",
    int(Keys.Key_Enter): "Enter",
    int(Keys.Key_Escape): "Escape",
    int(Keys.Key_F1): "F1",
    int(Keys.Key_F2): "F2",
    int(Keys.Key_F3): "F3",
    int(Keys.Key_F4): "F4",
    int(Keys.Key_F5): "F5",
    int(Keys.Key_F6): "F6",
    int(Keys.Key_F7): "F7",
    int(Keys.Key_F8): "F8",
    int(Keys.Key_F9): "F9",
    int(Keys.Key_F10): "F10",
    int(Keys.Key_F11): "F11",
    int(Keys.Key_F12): "F12",
    int(Keys.Key_Home): "Home",
    int(Keys.Key_Insert): "Insert",
    int(Keys.Key_Alt): "Alt",
    int(Keys.Key_Control): "Control",
    int(Keys.Key_Shift): "Shift",
    int(Keys.Key_Meta): "Meta",  # meta maps to control in QT on macOS, and vice-versa
    int(Keys.Key_NumLock): "NumLock",
    int(Keys.Key_PageDown): "PageDown",
    int(Keys.Key_PageUp): "Pageup",
    int(Keys.Key_Pause): "Pause",
    int(Keys.Key_ScrollLock): "ScrollLock",
    int(Keys.Key_Tab): "Tab",
}


def enable_hidpi():
    """Enable high-res displays."""
    set_dpi_aware = qt_version_info < (6, 4)  # Pyside
    if set_dpi_aware:
        try:
            # See https://github.com/pyzo/pyzo/pull/700 why we seem to need both
            # See https://github.com/pygfx/pygfx/issues/368 for high Qt versions
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

_show_image_method_warning = (
    "Qt falling back to offscreen rendering, which is less performant."
)


class QWgpuWidget(WgpuCanvasBase, QtWidgets.QWidget):
    """A QWidget representing a wgpu canvas that can be embedded in a Qt application."""

    def __init__(self, *args, present_method=None, **kwargs):
        super().__init__(*args, **kwargs)

        # Determine present method
        self._surface_ids = None
        if not present_method:
            self._present_to_screen = True
            if SYSTEM_IS_WAYLAND:
                # Trying to render to screen on Wayland segfaults. This might be because
                # the "display" is not the real display id. We can tell Qt to use
                # XWayland, so we can use the X11 path. This worked at some point,
                # but later this resulted in a Rust panic. So, until this is sorted
                # out, we fall back to rendering via an image.
                self._present_to_screen = False
        elif present_method == "screen":
            self._present_to_screen = True
        elif present_method == "image":
            self._present_to_screen = False
        else:
            raise ValueError(f"Invalid present_method {present_method}")

        self._is_closed = False

        self.setAttribute(WA_PaintOnScreen, self._present_to_screen)
        self.setAutoFillBackground(False)
        self.setAttribute(WA_DeleteOnClose, True)
        self.setAttribute(WA_InputMethodEnabled, True)
        self.setMouseTracking(True)
        self.setFocusPolicy(FocusPolicy.StrongFocus)

    def paintEngine(self):  # noqa: N802 - this is a Qt method
        # https://doc.qt.io/qt-5/qt.html#WidgetAttribute-enum  WA_PaintOnScreen
        if self._present_to_screen:
            return None
        else:
            return super().paintEngine()

    def paintEvent(self, event):  # noqa: N802 - this is a Qt method
        self._draw_frame_and_present()

    # Methods that we add from wgpu (snake_case)

    def _request_draw(self):
        # Ask Qt to do a paint event
        QtWidgets.QWidget.update(self)

    def _force_draw(self):
        # Call the paintEvent right now.
        # This works on all platforms I tested, except on MacOS when drawing with the 'image' method.
        # Not sure why this is. It be made to work by calling processEvents() but that has all sorts
        # of nasty side-effects (e.g. the scheduler timer keeps ticking, invoking other draws, etc.).
        self.repaint()

    def _get_loop(self):
        return loop

    def _get_surface_ids(self):
        if sys.platform.startswith("win") or sys.platform.startswith("darwin"):
            return {
                "window": int(self.winId()),
            }
        elif sys.platform.startswith("linux"):
            if False:
                # We fall back to XWayland, see _gui_utils.py
                return {
                    "platform": "wayland",
                    "window": int(self.winId()),
                    "display": int(get_alt_wayland_display()),
                }
            else:
                return {
                    "platform": "x11",
                    "window": int(self.winId()),
                    "display": int(get_alt_x11_display()),
                }
        else:
            raise RuntimeError(f"Cannot get Qt surface info on {sys.platform}.")

    def get_present_info(self):
        global _show_image_method_warning
        if self._surface_ids is None:
            self._surface_ids = self._get_surface_ids()
        if self._present_to_screen:
            info = {"method": "screen"}
            info.update(self._surface_ids)
        else:
            if _show_image_method_warning:
                logger.warn(_show_image_method_warning)
                _show_image_method_warning = None
            info = {
                "method": "image",
                "formats": ["rgba8unorm-srgb", "rgba8unorm"],
            }
        return info

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
        parent = self.parent()
        if isinstance(parent, QWgpuCanvas):
            parent.resize(width, height)
        else:
            self.resize(width, height)  # See comment on pixel ratio

    def _set_title(self, title):
        # A QWidgets title can actually be shown when the widget is shown in a dock.
        # But the application should probably determine that title, not us.
        parent = self.parent()
        if isinstance(parent, QWgpuCanvas):
            parent.setWindowTitle(title)

    def close(self):
        QtWidgets.QWidget.close(self)

    def is_closed(self):
        return self._is_closed

    # User events to jupyter_rfb events

    def _key_event(self, event_type, event):
        modifiers = tuple(
            MODIFIERS_MAP[mod]
            for mod in MODIFIERS_MAP.keys()
            if mod & event.modifiers()
        )

        ev = {
            "event_type": event_type,
            "key": KEY_MAP.get(event.key(), event.text()),
            "modifiers": modifiers,
        }
        self.submit_event(ev)

    def _char_input_event(self, char_str):
        ev = {
            "event_type": "char",
            "char_str": char_str,
            "modifiers": None,
        }
        self.submit_event(ev)

    def keyPressEvent(self, event):  # noqa: N802
        self._key_event("key_down", event)
        self._char_input_event(event.text())

    def keyReleaseEvent(self, event):  # noqa: N802
        self._key_event("key_up", event)

    def inputMethodEvent(self, event):  # noqa: N802
        commit_string = event.commitString()
        if commit_string:
            self._char_input_event(commit_string)

    def _mouse_event(self, event_type, event, touches=True):
        button = BUTTON_MAP.get(event.button(), 0)
        buttons = tuple(
            BUTTON_MAP[button]
            for button in BUTTON_MAP.keys()
            if button & event.buttons()
        )

        # For Qt on macOS Control and Meta are switched
        modifiers = tuple(
            MODIFIERS_MAP[mod]
            for mod in MODIFIERS_MAP.keys()
            if mod & event.modifiers()
        )

        ev = {
            "event_type": event_type,
            "x": event.pos().x(),
            "y": event.pos().y(),
            "button": button,
            "buttons": buttons,
            "modifiers": modifiers,
        }
        if touches:
            ev.update(
                {
                    "ntouches": 0,
                    "touches": {},  # TODO: Qt touch events
                }
            )

        self.submit_event(ev)

    def mousePressEvent(self, event):  # noqa: N802
        self._mouse_event("pointer_down", event)

    def mouseMoveEvent(self, event):  # noqa: N802
        self._mouse_event("pointer_move", event)

    def mouseReleaseEvent(self, event):  # noqa: N802
        self._mouse_event("pointer_up", event)

    def mouseDoubleClickEvent(self, event):  # noqa: N802
        super().mouseDoubleClickEvent(event)
        self._mouse_event("double_click", event, touches=False)

    def wheelEvent(self, event):  # noqa: N802
        # For Qt on macOS Control and Meta are switched
        modifiers = tuple(
            MODIFIERS_MAP[mod]
            for mod in MODIFIERS_MAP.keys()
            if mod & event.modifiers()
        )
        buttons = tuple(
            BUTTON_MAP[button]
            for button in BUTTON_MAP.keys()
            if button & event.buttons()
        )

        ev = {
            "event_type": "wheel",
            "dx": -event.angleDelta().x(),
            "dy": -event.angleDelta().y(),
            "x": event.position().x(),
            "y": event.position().y(),
            "buttons": buttons,
            "modifiers": modifiers,
        }
        self.submit_event(ev)

    def resizeEvent(self, event):  # noqa: N802
        ev = {
            "event_type": "resize",
            "width": float(event.size().width()),
            "height": float(event.size().height()),
            "pixel_ratio": self.get_pixel_ratio(),
        }
        self.submit_event(ev)

    def closeEvent(self, event):  # noqa: N802
        self._is_closed = True
        self.submit_event({"event_type": "close"})

    # Methods related to presentation of resulting image data

    def present_image(self, image_data, **kwargs):
        size = image_data.shape[1], image_data.shape[0]  # width, height
        rect1 = QtCore.QRect(0, 0, size[0], size[1])
        rect2 = self.rect()

        painter = QtGui.QPainter(self)
        # backingstore = self.backingStore()
        # backingstore.beginPaint(rect2)
        # painter = QtGui.QPainter(backingstore.paintDevice())

        # We want to simply blit the image (copy pixels one-to-one on framebuffer).
        # Maybe Qt does this when the sizes match exactly (like they do here).
        # Converting to a QPixmap and painting that only makes it slower.

        # Just in case, set render hints that may hurt performance.
        painter.setRenderHints(
            painter.RenderHint.Antialiasing | painter.RenderHint.SmoothPixmapTransform,
            False,
        )

        image = QtGui.QImage(
            image_data,
            size[0],
            size[1],
            size[0] * 4,
            QtGui.QImage.Format.Format_RGBA8888,
        )

        painter.drawImage(rect2, image, rect1)

        # Uncomment for testing purposes
        # painter.setPen(QtGui.QColor("#0000ff"))
        # painter.setFont(QtGui.QFont("Arial", 30))
        # painter.drawText(100, 100, "This is an image")

        painter.end()
        # backingstore.endPaint()
        # backingstore.flush(rect2)


class QWgpuCanvas(WgpuCanvasBase, QtWidgets.QWidget):
    """A toplevel Qt widget providing a wgpu canvas."""

    # Most of this is proxying stuff to the inner widget.
    # We cannot use a toplevel widget directly, otherwise the window
    # size can be set to subpixel (logical) values, without being able to
    # detect this. See https://github.com/pygfx/wgpu-py/pull/68

    def __init__(self, *, size=None, title=None, **kwargs):
        # When using Qt, there needs to be an
        # application before any widget is created
        loop.init_qt()

        sub_kwargs = pop_kwargs_for_base_canvas(kwargs)
        super().__init__(**kwargs)

        # Handle inputs
        if title is None:
            title = "qt canvas"
        if not size:
            size = 640, 480

        self.setAttribute(WA_DeleteOnClose, True)
        self.setMouseTracking(True)

        self._subwidget = QWgpuWidget(self, **sub_kwargs)
        self._events = self._subwidget._events

        # Note: At some point we called `self._subwidget.winId()` here. For some
        # reason this was needed to "activate" the canvas. Otherwise the viz was
        # not shown if no canvas was provided to request_adapter(). Removed
        # later because could not reproduce.

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        layout.addWidget(self._subwidget)

        self.set_logical_size(*size)
        self.set_title(title)
        self.show()

    # Qt methods

    def closeEvent(self, event):  # noqa: N802
        self._subwidget._is_closed = True
        self.submit_event({"event_type": "close"})

    # Methods that we add from wgpu (snake_case)

    def _request_draw(self):
        self._subwidget._request_draw()

    def _force_draw(self):
        self._subwidget._force_draw()

    def _get_loop(self):
        return None  # This means this outer widget won't have a scheduler

    def get_present_info(self):
        return self._subwidget.get_present_info()

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

    def close(self):
        QtWidgets.QWidget.close(self)

    def is_closed(self):
        return self._subwidget.is_closed()

    # Methods that we need to explicitly delegate to the subwidget

    def set_title(self, *args):
        self._subwidget.set_title(*args)

    def get_context(self, *args, **kwargs):
        return self._subwidget.get_context(*args, **kwargs)

    def request_draw(self, *args, **kwargs):
        return self._subwidget.request_draw(*args, **kwargs)

    def present_image(self, image, **kwargs):
        return self._subwidget.present_image(image, **kwargs)


# Make available under a name that is the same for all gui backends
WgpuWidget = QWgpuWidget
WgpuCanvas = QWgpuCanvas


class QtWgpuTimer(WgpuTimer):
    """Wgpu timer basef on Qt."""

    def _init(self):
        self._qt_timer = QtCore.QTimer()
        self._qt_timer.timeout.connect(self._tick)
        self._qt_timer.setSingleShot(True)
        self._qt_timer.setTimerType(PreciseTimer)

    def _start(self):
        self._qt_timer.start(int(self._interval * 1000))

    def _stop(self):
        self._qt_timer.stop()


class QtWgpuLoop(WgpuLoop):
    _TimerClass = QtWgpuTimer

    def init_qt(self):
        _ = self._app
        self._latest_timeout = 0

    @property
    def _app(self):
        """Return global instance of Qt app instance or create one if not created yet."""
        return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def _call_soon(self, callback, *args):
        func = callback
        if args:
            func = lambda: callback(*args)
        QtCore.QTimer.singleshot(0, func)

    def _run(self):
        # Note: we could detect if asyncio is running (interactive session) and wheter
        # we can use QtAsyncio. However, there's no point because that's up for the
        # end-user to decide.

        # Note: its possible, and perfectly ok, if the application is started from user
        # code. This works fine because the application object is global. This means
        # though, that we cannot assume anything based on whether this method is called
        # or not.

        if already_had_app_on_import:
            return  # Likely in an interactive session or larger application that will start the Qt app.

        app = self._app
        app.setQuitOnLastWindowClosed(False)
        app.exec() if hasattr(app, "exec") else app.exec_()

    def _stop(self):
        if not already_had_app_on_import:
            self._app.quit()

    def _wgpu_gui_poll(self):
        pass  # we assume the Qt event loop is running. Calling processEvents() will cause recursive repaints.


loop = QtWgpuLoop()
run = loop.run  # backwards compat
