"""
Support for rendering in a wxPython window. Provides a widget that
can be used as a standalone window or in a larger GUI.
"""

import ctypes

from .base import WgpuCanvasBase

import wx


try:
    # fix blurry text on windows
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass  # fail on non-windows


class WxWgpuCanvas(WgpuCanvasBase):
    """Base wx canvas class. Instantiating this will produce either
    the Window or Frame subclass flavor.
    """

    # I'd love this to inherit from wx.Window. I tried, but wx seems
    # to get confused by the complex class inheritance or something.

    def __new__(cls, parent=None, *args, **kwargs):
        parent = parent or kwargs.get("parent", None)
        if parent is None:
            return wx.Frame.__new__(WxWgpuFrame, *args, **kwargs)
        else:
            return wx.Window.__new__(WxWgpuWindow, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_ERASE_BACKGROUND, lambda x: None)

    def on_paint(self, event):
        dc = wx.PaintDC(self)  # needed for wx
        self._draw_frame_and_present()
        del dc
        event.Skip()

    def set_logical_size(self, width, height):
        if width < 0 or height < 0:
            raise ValueError("Window width and height must not be negative")
        self.SetSize(width, height)

    def get_logical_size(self):
        lsize = self.Size[0], self.Size[1]
        return float(lsize[0]), float(lsize[1])

    def close(self):
        self.Hide()

    def is_closed(self):
        return not self.IsShown()


class WxWgpuWindow(WxWgpuCanvas, wx.Window):
    """A wx widget providing a wgpu canvas."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Bind(wx.EVT_SIZE, lambda e: self._request_draw())

    def get_window_id(self):
        return int(self.GetHandle())

    def get_pixel_ratio(self):
        # todo: this is not hidpi-ready (at least on win10)
        # Observations:
        # * On Win10 this always returns 1 - so hidpi is effectively broken
        return self.GetContentScaleFactor()

    def get_physical_size(self):
        lsize = self.Size[0], self.Size[1]
        lsize = float(lsize[0]), float(lsize[1])
        ratio = self.GetContentScaleFactor()
        return round(lsize[0] * ratio), round(lsize[1] * ratio)

    def _request_draw(self):
        # todo: this does the draw *directly* which is not what we want
        # e.g. it would cause recursion errors in most pygfx examples
        self.Refresh()  # Invalidates the canvas
        self.Update()  # Redraw


class WxWgpuFrame(WxWgpuCanvas, wx.Frame):
    """A wx frame providing a wgpu canvas."""

    # Most of this is proxying stuff to the inner widget

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        size = kwargs.pop("size", None) or (640, 480)
        self.set_logical_size(*size)

        self.canvas = WxWgpuWindow(parent=self)
        self.Bind(wx.EVT_CLOSE, lambda e: self.Destroy())

        self.Show()

    def get_window_id(self):
        return self.canvas.get_window_id()

    def get_pixel_ratio(self):
        return self.canvas.get_pixel_ratio()

    def get_physical_size(self):
        return self.canvas.get_physical_size()

    def request_draw(self, func=None):
        return self.canvas.request_draw(func)

    def close(self):
        self.Hide()

    def is_closed(self):
        return not self.IsShown()


# Make available under a name that is the same for all gui backends
WgpuCanvas = WxWgpuCanvas
