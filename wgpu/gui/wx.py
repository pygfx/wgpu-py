"""
Support for rendering in a wxPython window. Provides a widget that
can be used as a standalone window or in a larger GUI.
"""

import sys
import ctypes

from .base import WgpuCanvasBase
from ._gui_utils import get_alt_x11_display, get_alt_wayland_display, weakbind

import wx


is_wayland = False  # We force wx to use X11 in _gui_utils.py


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


class WxWgpuWindow(WgpuCanvasBase, wx.Window):
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

    def on_paint(self, event):
        dc = wx.PaintDC(self)  # needed for wx
        if not self._draw_lock:
            self._draw_frame_and_present()
        del dc
        event.Skip()

    def _on_resize(self, *args):
        self._draw_lock = True
        self._resize_timer.Start(100, wx.TIMER_ONE_SHOT)

    def _on_resize_done(self, *args):
        self._draw_lock = False
        self._request_draw()

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

    def _request_draw(self):
        # Despite the FPS limiting the delayed call to refresh solves
        # that drawing only happens when the mouse is down, see #209.
        if not self._request_draw_timer.IsRunning():
            self._request_draw_timer.Start(
                self._get_draw_wait_time() * 1000, wx.TIMER_ONE_SHOT
            )

    def close(self):
        self.Hide()

    def is_closed(self):
        return not self.IsShown()


class WxWgpuCanvas(WgpuCanvasBase, wx.Frame):
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


# Make available under a name that is the same for all gui backends
WgpuWidget = WxWgpuWindow
WgpuCanvas = WxWgpuCanvas
