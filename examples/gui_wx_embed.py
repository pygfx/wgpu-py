"""
An example demonstrating a wx app with a wgpu viz inside.
"""

# run_example = false

import wx
from rendercanvas.wx import WxRenderWidget

from triangle import setup_drawing_sync


class Example(wx.Frame):
    def __init__(self):
        super().__init__(None, title="wgpu triangle embedded in a wx app")
        self.SetSize(640, 480)

        splitter = wx.SplitterWindow(self)

        self.button = wx.Button(self, -1, "Hello world")
        self.canvas1 = WxRenderWidget(splitter)
        self.canvas2 = WxRenderWidget(splitter)

        splitter.SplitVertically(self.canvas1, self.canvas2)
        splitter.SetSashGravity(0.5)

        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.button, 0, wx.EXPAND)
        sizer.Add(splitter, 1, wx.EXPAND)
        self.SetSizer(sizer)

        self.Show()


app = wx.App()
example = Example()

draw_frame1 = setup_drawing_sync(example.canvas1)
draw_frame2 = setup_drawing_sync(example.canvas2)

example.canvas1.request_draw(draw_frame1)
example.canvas2.request_draw(draw_frame2)

app.MainLoop()
