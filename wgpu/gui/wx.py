"""
Support for rendering in a wxPython window. Provides a widget that
can be used as a standalone window or in a larger GUI.
"""

import ctypes
import sys
from typing import Optional

import wx

from ._gui_utils import get_alt_x11_display, get_alt_wayland_display, weakbind
from .base import WgpuCanvasBase, WgpuAutoGui

is_wayland = False  # We force wx to use X11 in _gui_utils.py

BUTTON_MAP = {
    wx.MOUSE_BTN_LEFT: 1,
    wx.MOUSE_BTN_RIGHT: 2,
    wx.MOUSE_BTN_MIDDLE: 3,
    wx.MOUSE_BTN_AUX1: 4,
    wx.MOUSE_BTN_AUX2: 5,
    # wxPython doesn't have exact equivalents for TaskButton, ExtraButton4, and ExtraButton5
}

MOUSE_EVENT_MAP = {
    "pointer_down": [
        wx.wxEVT_LEFT_DOWN,
        wx.wxEVT_MIDDLE_DOWN,
        wx.wxEVT_RIGHT_DOWN,
        wx.wxEVT_AUX1_DOWN,
        wx.wxEVT_AUX2_DOWN,
    ],
    "pointer_up": [
        wx.wxEVT_LEFT_UP,
        wx.wxEVT_MIDDLE_UP,
        wx.wxEVT_RIGHT_UP,
        wx.wxEVT_AUX1_UP,
        wx.wxEVT_AUX2_UP,
    ],
    "double_click": [
        wx.wxEVT_LEFT_DCLICK,
        wx.wxEVT_MIDDLE_DCLICK,
        wx.wxEVT_RIGHT_DCLICK,
        wx.wxEVT_AUX1_DCLICK,
        wx.wxEVT_AUX2_DCLICK,
    ],
    "wheel": [wx.wxEVT_MOUSEWHEEL],
}

# reverse the mouse event map (from one-to-many to many-to-one)
MOUSE_EVENT_MAP_REVERSED = {
    value: key for key, values in MOUSE_EVENT_MAP.items() for value in values
}

MODIFIERS_MAP = {
    wx.MOD_SHIFT: "Shift",
    wx.MOD_CONTROL: "Control",
    wx.MOD_ALT: "Alt",
    wx.MOD_META: "Meta",
}

KEY_MAP = {
    wx.WXK_DOWN: "ArrowDown",
    wx.WXK_UP: "ArrowUp",
    wx.WXK_LEFT: "ArrowLeft",
    wx.WXK_RIGHT: "ArrowRight",
    wx.WXK_BACK: "Backspace",
    wx.WXK_CAPITAL: "CapsLock",
    wx.WXK_DELETE: "Delete",
    wx.WXK_END: "End",
    wx.WXK_RETURN: "Enter",
    wx.WXK_ESCAPE: "Escape",
    wx.WXK_F1: "F1",
    wx.WXK_F2: "F2",
    wx.WXK_F3: "F3",
    wx.WXK_F4: "F4",
    wx.WXK_F5: "F5",
    wx.WXK_F6: "F6",
    wx.WXK_F7: "F7",
    wx.WXK_F8: "F8",
    wx.WXK_F9: "F9",
    wx.WXK_F10: "F10",
    wx.WXK_F11: "F11",
    wx.WXK_F12: "F12",
    wx.WXK_HOME: "Home",
    wx.WXK_INSERT: "Insert",
    wx.WXK_ALT: "Alt",
    wx.WXK_CONTROL: "Control",
    wx.WXK_SHIFT: "Shift",
    wx.WXK_COMMAND: "Meta",  # wx.WXK_COMMAND is used for Meta (Command key on macOS)
    wx.WXK_NUMLOCK: "NumLock",
    wx.WXK_PAGEDOWN: "PageDown",
    wx.WXK_PAGEUP: "PageUp",
    wx.WXK_PAUSE: "Pause",
    wx.WXK_SCROLL: "ScrollLock",
    wx.WXK_TAB: "Tab",
}


def enable_hidpi():
    """Enable high-res displays."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass  # fail on non-windows


enable_hidpi()


class TimerWithCallback(wx.Timer):
    def __init__(self, callback):
        super().__init__()
        self._callback = callback

    def Notify(self, *args):  # noqa: N802
        try:
            self._callback()
        except RuntimeError:
            pass  # wrapped C/C++ object of type WxWgpuWindow has been deleted


class WxWgpuWindow(WgpuAutoGui, WgpuCanvasBase, wx.Window):
    """A wx Window representing a wgpu canvas that can be embedded in a wx application."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # A timer for limiting fps
        self._request_draw_timer = TimerWithCallback(self.Refresh)

        # We keep a timer to prevent draws during a resize. This prevents
        # issues with mismatching present sizes during resizing (on Linux).
        self._resize_timer = TimerWithCallback(self._on_resize_done)
        self._draw_lock = False

        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_ERASE_BACKGROUND, lambda x: None)
        self.Bind(wx.EVT_SIZE, self._on_resize)

        self.Bind(wx.EVT_KEY_DOWN, self._on_key_down)
        self.Bind(wx.EVT_KEY_UP, self._on_key_up)

        self.Bind(wx.EVT_MOUSE_EVENTS, self._on_mouse_events)
        self.Bind(wx.EVT_MOTION, self._on_mouse_move)

    def on_paint(self, event):
        dc = wx.PaintDC(self)  # needed for wx
        if not self._draw_lock:
            self._draw_frame_and_present()
        del dc
        event.Skip()

    def _on_resize(self, event: wx.SizeEvent):
        self._draw_lock = True
        self._resize_timer.Start(100, wx.TIMER_ONE_SHOT)

        # fire resize event
        size: wx.Size = event.GetSize()
        ev = {
            "event_type": "resize",
            "width": float(size.GetWidth()),
            "height": float(size.GetHeight()),
            "pixel_ratio": self.get_pixel_ratio(),
        }
        self._handle_event_and_flush(ev)

    def _on_resize_done(self, *args):
        self._draw_lock = False
        self._request_draw()

    # Methods for input events

    def _on_key_down(self, event: wx.KeyEvent):
        char_str = self._get_char_from_event(event)
        self._key_event("key_down", event, char_str)

        if char_str is not None:
            self._char_input_event(char_str)

    def _on_key_up(self, event: wx.KeyEvent):
        char_str = self._get_char_from_event(event)
        self._key_event("key_up", event, char_str)

    def _key_event(self, event_type: str, event: wx.KeyEvent, char_str: Optional[str]):
        modifiers = tuple(
            MODIFIERS_MAP[mod]
            for mod in MODIFIERS_MAP.keys()
            if mod & event.GetModifiers()
        )

        ev = {
            "event_type": event_type,
            "key": KEY_MAP.get(event.GetKeyCode(), char_str),
            "modifiers": modifiers,
        }
        self._handle_event_and_flush(ev)

    def _char_input_event(self, char_str: Optional[str]):
        if char_str is None:
            return

        ev = {
            "event_type": "char",
            "char_str": char_str,
            "modifiers": None,
        }
        self._handle_event_and_flush(ev)

    @staticmethod
    def _get_char_from_event(event: wx.KeyEvent) -> Optional[str]:
        keycode = event.GetKeyCode()
        modifiers = event.GetModifiers()

        # Check if keycode corresponds to a printable ASCII character
        if 32 <= keycode <= 126:
            char = chr(keycode)
            if not modifiers & wx.MOD_SHIFT:
                char = char.lower()
            return char

        # Check for special keys (e.g., Enter, Tab)
        if keycode == wx.WXK_RETURN:
            return "\n"
        if keycode == wx.WXK_TAB:
            return "\t"

        # Handle non-ASCII characters and others
        uni_char = event.GetUnicodeKey()
        if uni_char != wx.WXK_NONE:
            return chr(uni_char)

        return None

    def _mouse_event(self, event_type: str, event: wx.MouseEvent, touches: bool = True):
        button = BUTTON_MAP.get(event.GetButton(), 0)
        buttons = (button,)  # in wx only one button is pressed per event

        modifiers = tuple(
            MODIFIERS_MAP[mod]
            for mod in MODIFIERS_MAP.keys()
            if mod & event.GetModifiers()
        )

        ev = {
            "event_type": event_type,
            "x": event.GetX(),
            "y": event.GetY(),
            "button": button,
            "buttons": buttons,
            "modifiers": modifiers,
        }

        if touches:
            ev.update(
                {
                    "ntouches": 0,
                    "touches": {},  # TODO: Wx touch events
                }
            )

        if event_type == "wheel":
            delta = event.GetWheelDelta()
            axis = event.GetWheelAxis()
            rotation = event.GetWheelRotation()

            dx = 0
            dy = 0

            if axis == wx.MOUSE_WHEEL_HORIZONTAL:
                dx = delta * rotation
            elif axis == wx.MOUSE_WHEEL_VERTICAL:
                dy = delta * rotation

            ev.update({"dx": -dx, "dy": -dy})

            match_keys = {"modifiers"}
            accum_keys = {"dx", "dy"}
            self._handle_event_rate_limited(
                ev, self._call_later, match_keys, accum_keys
            )
        elif event_type == "pointer_move":
            match_keys = {"buttons", "modifiers", "ntouches"}
            accum_keys = {}
            self._handle_event_rate_limited(
                ev, self._call_later, match_keys, accum_keys
            )
        else:
            self._handle_event_and_flush(ev)

    def _on_mouse_events(self, event: wx.MouseEvent):
        event_type = event.GetEventType()

        event_type_name = MOUSE_EVENT_MAP_REVERSED.get(event_type, None)
        if event_type_name is None:
            return

        self._mouse_event(event_type_name, event)

    def _on_mouse_move(self, event: wx.MouseEvent):
        self._mouse_event("pointer_move", event)

    # Methods that we add from wgpu

    def get_surface_info(self):
        if sys.platform.startswith("win") or sys.platform.startswith("darwin"):
            return {
                "window": int(self.GetHandle()),
            }
        elif sys.platform.startswith("linux"):
            if is_wayland:
                return {
                    "platform": "wayland",
                    "window": int(self.GetHandle()),
                    "display": int(get_alt_wayland_display()),
                }
            else:
                return {
                    "platform": "x11",
                    "window": int(self.GetHandle()),
                    "display": int(get_alt_x11_display()),
                }
        else:
            raise RuntimeError(f"Cannot get Qt surafce info on {sys.platform}.")

    def get_pixel_ratio(self):
        # todo: this is not hidpi-ready (at least on win10)
        # Observations:
        # * On Win10 this always returns 1 - so hidpi is effectively broken
        return self.GetContentScaleFactor()

    def get_logical_size(self):
        lsize = self.Size[0], self.Size[1]
        return float(lsize[0]), float(lsize[1])

    def get_physical_size(self):
        lsize = self.Size[0], self.Size[1]
        lsize = float(lsize[0]), float(lsize[1])
        ratio = self.GetContentScaleFactor()
        return round(lsize[0] * ratio + 0.01), round(lsize[1] * ratio + 0.01)

    def set_logical_size(self, width, height):
        if width < 0 or height < 0:
            raise ValueError("Window width and height must not be negative")
        self.SetSize(width, height)

    def set_title(self, title):
        pass  # only on frames

    def _request_draw(self):
        # Despite the FPS limiting the delayed call to refresh solves
        # that drawing only happens when the mouse is down, see #209.
        if not self._request_draw_timer.IsRunning():
            self._request_draw_timer.Start(
                max(1, int(self._get_draw_wait_time() * 1000)), wx.TIMER_ONE_SHOT
            )

    def close(self):
        self.Hide()

    def is_closed(self):
        return not self.IsShown()

    @staticmethod
    def _call_later(delay, callback, *args):
        delay_ms = int(delay * 1000)
        if delay_ms <= 0:
            callback(*args)

        wx.CallLater(max(delay_ms, 1), callback, *args)


class WxWgpuCanvas(WgpuAutoGui, WgpuCanvasBase, wx.Frame):
    """A toplevel wx Frame providing a wgpu canvas."""

    # Most of this is proxying stuff to the inner widget.

    def __init__(self, *, parent=None, size=None, title=None, max_fps=30, **kwargs):
        super().__init__(parent, **kwargs)

        self.set_logical_size(*(size or (640, 480)))
        self.SetTitle(title or "wx wgpu canvas")

        self._subwidget = WxWgpuWindow(parent=self, max_fps=max_fps)
        self._subwidget.add_event_handler(weakbind(self.handle_event), "*")
        self.Bind(wx.EVT_CLOSE, lambda e: self.Destroy())

        self.Show()

    # wx methods

    def Refresh(self):  # noqa: N802
        super().Refresh()
        self._subwidget.Refresh()

    # Methods that we add from wgpu

    def get_surface_info(self):
        return self._subwidget.get_surface_info()

    def get_pixel_ratio(self):
        return self._subwidget.get_pixel_ratio()

    def get_logical_size(self):
        return self._subwidget.get_logical_size()

    def get_physical_size(self):
        return self._subwidget.get_physical_size()

    def set_logical_size(self, width, height):
        if width < 0 or height < 0:
            raise ValueError("Window width and height must not be negative")
        self.SetSize(width, height)

    def set_title(self, title):
        self.SetTitle(title)

    def _request_draw(self):
        return self._subwidget._request_draw()

    def close(self):
        self._handle_event_and_flush({"event_type": "close"})
        super().close()

    def is_closed(self):
        return not self.isVisible()

    # Methods that we need to explicitly delegate to the subwidget

    def get_context(self, *args, **kwargs):
        return self._subwidget.get_context(*args, **kwargs)

    def request_draw(self, *args, **kwargs):
        return self._subwidget.request_draw(*args, **kwargs)


# Make available under a name that is the same for all gui backends
WgpuWidget = WxWgpuWindow
WgpuCanvas = WxWgpuCanvas
