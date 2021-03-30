"""
Support for rendering in a wxPython window. Provides a widget that
can be used as a standalone window or in a larger GUI.
"""

import math
import ctypes

from .base import WgpuCanvasBase

import wx

try:
    # fix blurry text on windows
    ctypes.windll.shcore.SetProcessDpiAwareness(True)
except Exception:
    pass  # fail on non-windows


class WxWgpuCanvas(WgpuCanvasBase, wx.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_ERASE_BACKGROUND, lambda x: None)

    def on_paint(self, event):
        dc = wx.PaintDC(self)  # needed for wx
        self._draw_frame_and_present()
        del dc
        event.Skip()

    def get_window_id(self):
        return int(self.GetHandle())

    def get_pixel_ratio(self):
        return self.GetContentScaleFactor()

    def get_logical_size(self):
        lsize = self.Size[0], self.Size[1]
        return float(lsize[0]), float(lsize[1])

    def get_physical_size(self):
        lsize = self.Size[0], self.Size[1]
        lsize = float(lsize[0]), float(lsize[1])
        ratio = self.GetContentScaleFactor()
        return math.ceil(lsize[0] * ratio), math.ceil(lsize[1] * ratio)

    def set_logical_size(self, width, height):
        if width < 0 or height < 0:
            raise ValueError("Window width and height must not be negative")
        self.SetSize(width, height)

    def _request_draw(self):
        pass

    def close(self):
        self.Hide()

    def is_closed(self):
        return not self.IsShown()


WgpuCanvas = WxWgpuCanvas
