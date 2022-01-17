from collections import defaultdict
import importlib
import sys
import traceback

for libname in ("PySide6", "PyQt6", "PySide2", "PyQt5"):
    try:
        importlib.import_module(libname)
        break
    except ModuleNotFoundError:
        pass

from ..qt import WgpuCanvas, QtCore, QtWidgets  # noqa: E402

# Global reference to an app. Will be either instantiated or set to the
# global Qt instance when the first QAutoWgpuCanvas object is created
app = None


def run():
    app.exec() if hasattr(app, "exec") else app.exec_()


def call_later(delay, callback, *args):
    QtCore.QTimer.singleShot(delay * 1000, lambda: callback(*args))


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


class QAutoWgpuCanvas(WgpuCanvas):
    def __init__(self, *args, **kwargs):
        # When using Qt, there needs to be an
        # application before any widget is created
        global app
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        super().__init__(*args, **kwargs)

        self._event_handlers = defaultdict(set)

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

    def handle_event(self, event):
        """Handle an incoming event.

        Subclasses can overload this method. Events include widget
        resize, mouse/touch interaction, key events, and more. An event
        is a dict with at least the key event_type. For details, see
        https://jupyter-rfb.readthedocs.io/en/latest/events.html
        """
        event_type = event.get("event_type")
        for callback in self._event_handlers[event_type]:
            callback(event)

    def add_event_handler(self, *args):
        """Register an event handler.

        Arguments:
            callback (callable): The event handler. Must accept a
                single event argument.
            *types (list of strings): A list of event types.

        For the available events, see
        https://jupyter-rfb.readthedocs.io/en/latest/events.html

        Can also be used as a decorator.

        Example:

        .. code-block:: py

            def my_handler(event):
                print(event)

            canvas.add_event_handler(my_handler, "pointer_up", "pointer_down")

        Decorator usage example:

        .. code-block:: py

            @canvas.add_event_handler("pointer_up", "pointer_down")
            def my_handler(event):
                print(event)
        """
        decorating = not callable(args[0])
        callback = None if decorating else args[0]
        types = args if decorating else args[1:]

        def decorator(_callback):
            for type in types:
                self._event_handlers[type].add(_callback)
            return _callback

        if decorating:
            return decorator
        return decorator(callback)

    def remove_event_handler(self, callback, *types):
        """Unregister an event handler.

        Arguments:
            callback (callable): The event handler.
            *types (list of strings): A list of event types.
        """
        for type in types:
            self._event_handlers[type].remove(callback)

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


WgpuCanvas = QAutoWgpuCanvas
