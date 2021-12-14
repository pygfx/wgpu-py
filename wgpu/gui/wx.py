"""
Support for rendering in a wxPython window. Provides a widget that
can be used as a standalone window or in a larger GUI.
"""

import time
import ctypes

from .base import WgpuCanvasBase

import wx


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
        self._callback()


class WxWgpuWindow(WgpuCanvasBase, wx.Window):
    """A wx Window representing a wgpu canvas that can be embedded in a wx application."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Variables to limit the fps
        self._draw_time = 0
        self._target_fps = 30
        self._request_draw_timer = TimerWithCallback(self.Refresh)

        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_ERASE_BACKGROUND, lambda x: None)
        self.Bind(wx.EVT_SIZE, lambda e: self._request_draw())

    def on_paint(self, event):
        self._draw_time = time.perf_counter()
        dc = wx.PaintDC(self)  # needed for wx
        self._draw_frame_and_present()
        del dc
        event.Skip()

    # Methods that we add from wgpu

    def get_window_id(self):
        return int(self.GetHandle())

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
        # Despite the FPS limiting the delayed call to refresh solves two other issues:
        # * It prevents that drawing only happens when the mouse is down, see #209.
        # * It prevents issues with mismatching present sizes during resizing (on Linux).
        if not self._request_draw_timer.IsRunning():
            now = time.perf_counter()
            target_time = self._draw_time + 1 / self._target_fps
            wait_time = max(0, target_time - now)
            self._request_draw_timer.Start(wait_time * 1000, wx.TIMER_ONE_SHOT)

    def close(self):
        self.Hide()

    def is_closed(self):
        return not self.IsShown()


class WxWgpuCanvas(WgpuCanvasBase, wx.Frame):
    """A toplevel wx Frame providing a wgpu canvas."""

    # Most of this is proxying stuff to the inner widget.

    def __init__(self, *, parent=None, size=None, title=None, **kwargs):
        super().__init__(parent, **kwargs)

        self.set_logical_size(*(size or (640, 480)))
        self.SetTitle(title or "wx wgpu canvas")

        self._subwidget = WxWgpuWindow(parent=self)
        self.Bind(wx.EVT_CLOSE, lambda e: self.Destroy())

        self.Show()

    # wx methods

    def Refresh(self):  # noqa: N802
        super().Refresh()
        self._subwidget.Refresh()

    # Methods that we add from wgpu

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
