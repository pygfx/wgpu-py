"""
Import the viz from triangle.py and run it in a wxPython window.
"""

import wx
from wgpu.gui.wxpython import WgpuCanvas
import wgpu.backends.rs  # noqa: F401, Select Rust backend

# Import the (async) function that we must call to run the visualization
from examples.triangle import main


class AppFrame(wx.Frame):
    def __init__(self, *args, **kwargs):
        wx.Frame.__init__(self, *args, **kwargs)

        self.canvas = WgpuCanvas(self)

        self.Bind(wx.EVT_CLOSE, self.OnDestroy)

    def OnDestroy(self, event):
        self.Destroy()

app = wx.App()
frm = AppFrame(parent=None, title="wgpu triangle with wxPython")
frm.Show()
main(frm.canvas)
app.MainLoop()
