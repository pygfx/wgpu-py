"""
Support to render in a Tkinter window. Tk is a bit of a pain to work
with (its ancient stuff), but it's always there, no extra dependencies.
"""

import tkinter

from wgpu.gui.base import BaseCanvas


class TkWgpuCanvas(BaseCanvas, tkinter.Toplevel):
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

    def get_size_and_pixel_ratio(self):
        width = self.winfo_width()
        height = self.winfo_height()
        pixelratio = 1  # todo: pixelratio? Ask for it via win32 api?
        return width, height, pixelratio

    def is_closed(self):
        try:
            return self.wm_state() != "normal"
        except Exception:
            return True

    def get_display_id(self):
        return super().get_display_id()  # uses X11 lib

    def get_window_id(self):
        # There's two functions of interest here. On Linux they return the same
        # value (for a toplevel widget), but on Windows not, and frame() does
        # not work.
        # * self.winfo_id(): platform-specific identifier for window
        # * self.frame(): platform specific window identifier for the
        #   outermost frame that contains window
        return int(self.winfo_id())
        # return int(self.frame(), 16)

    def _paint(self, *args):
        # Actual drawing needs to happen *after* Tcl draws bg
        self.after(1, self._draw_frame_and_present)


WgpuCanvas = TkWgpuCanvas
