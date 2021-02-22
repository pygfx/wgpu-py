"""
Support for rendering in a wxPython window. Provides a widget that
can be used as a standalone window or in a larger GUI.
"""

import sys
import time
import ctypes
import importlib

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

    def get_window_id(self):
        return int(self.GetId())

    def get_pixel_ratio(self):
        return self.GetDPIScaleFactor()

    def get_logical_size(self):
        lsize = self.GetWidth(), self.GetHeight()
        return float(lsize[0]), float(lsize[1])

    def get_physical_size(self):
        lsize = self.GetWidth(), self.GetHeight()
        lsize = float(lsize[0]), float(lsize[1])
        ratio = self.GetDPIScaleFactor()
        return round(lsize[0] * ratio), round(lsize[1] * ratio)

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
