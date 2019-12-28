"""
Support to render in a Tkinter window. Tk is a bit of a pain to work
with (its ancient stuff), but it's always there, no extra dependencies.
"""

import tkinter

from wgpu.gui.base import BaseCanvas


class WgpuCanvas(BaseCanvas, tkinter.Toplevel):
    """ A canvas object base on a Tkinter TopLevel window.
    """

    def __init__(self, *args, size=None, title=None, **kwargs):
        super().__init__(*args, **kwargs)

        if size:
            width, height = size
            self.geometry(f"{int(width)}x{int(height)}")
        if title:
            self.title(title)

        # Listen for events for the right time to draw the window.
        # See http://www.tcl.tk/man/tcl8.4/TkCmd/bind.htm#M7
        # You'd think Expose would do the trick, but Configure seems
        # to be the most important one. Enter is a backup (moving the
        # mouse into the window will draw it).
        for ev_name in ["<Expose>", "<Configure>", "<Enter>"]:
            self.bind(ev_name, self._paint)

    def getSizeAndPixelRatio(self):
        width = self.winfo_width()
        height = self.winfo_height()
        pixelratio = 1  # todo: pixelratio? Ask for it via win32 api?
        return width, height, pixelratio

    def isClosed(self):
        try:
            return self.wm_state() != "normal"
        except Exception:
            return True

    def getWindowId(self):
        return int(self.winfo_id())
        # The docs seem to say that the below also produces the native window id,
        # but the resulting int is different, and with it things don't work:
        # return int(self.frame(), 16)

    def _paint(self, *args):
        # Actual drawing needs to happen *after* Tcl draws bg
        self.after(1, self._drawFrameAndPresent)
