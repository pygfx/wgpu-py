import os
import sys
import logging
import ctypes.util

logger = logging.getLogger("wgpu")


class WgpuCanvasInterface:
    """ This is the interface that a canvas object must implement in order
    to be a valid canvas that wgpu can work with.
    """

    # NOTE: It is not necessary to actually subclass this class.

    def __init__(self, *args, **kwargs):
        # The args/kwargs are there because we may be mixed with e.g. a Qt widget
        super().__init__(*args, **kwargs)
        self._display_id = None

    def get_window_id(self):
        """ Get the native window id. This is used by the backends
        to obtain a surface id.
        """
        raise NotImplementedError()

    def get_display_id(self):
        """ Get the native display id on Linux. This is used by the backends
        to obtain a surface id on Linux. The default implementation calls into
        the X11 lib to get the display id.
        """
        # Re-use to avoid creating loads of id's
        if self._display_id is not None:
            return self._display_id

        if sys.platform.startswith("linux"):
            is_wayland = "wayland" in os.getenv("XDG_SESSION_TYPE", "").lower()
            if is_wayland:
                raise NotImplementedError(
                    f"Cannot (yet) get display id on {self.__class__.__name__}."
                )
            else:
                x11 = ctypes.CDLL(ctypes.util.find_library("X11"))
                x11.XOpenDisplay.restype = ctypes.c_void_p
                self._display_id = x11.XOpenDisplay(None)
        else:
            raise RuntimeError(f"Cannot get display id on {sys.platform}.")

        return self._display_id

    def get_physical_size(self):
        """ Get the physical size in integer pixels.
        """
        raise NotImplementedError()


class WgpuCanvasBase(WgpuCanvasInterface):
    """ An abstract class, extending :class:`WgpuCanvasInterface`,
    that provides a base canvas for various GUI toolkits, so
    that basic canvas functionality is available via a common API.

    It convenient, but not required to use this class (or any of its
    subclasses) to use wgpu-py.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._err_hashes = {}

    def draw_frame(self):
        """ The function that gets called at each draw. You can implement
        this method in a subclass, or assign the attribute directly.
        """
        pass

    def _draw_frame_and_present(self):
        """ Draw the frame and present the swapchain. Errors are logged to the
        "wgpu" logger. Should be called by the subclass at an appropriate time.
        """
        # Perform the user-defined drawing code. When this errors,
        # we should report the error and then continue, otherwise we crash.
        try:
            self.draw_frame()
        except Exception as err:
            self._log_exception("Draw error", err)

    def _log_exception(self, kind, err):
        """ Log the given exception instance, but only log a one-liner for
        subsequent occurances of the same error to avoid spamming (which
        can happen easily with errors in the drawing code).
        """
        msg = str(err)
        msgh = hash(msg)
        if msgh not in self._err_hashes:
            # Provide the exception, so the default logger prints a stacktrace.
            # IDE's can get the exception from the root logger for PM debugging.
            self._err_hashes[msgh] = 1
            logger.error(kind, exc_info=err)
        else:
            # We've seen this message before, return a one-liner instead.
            self._err_hashes[msgh] = count = self._err_hashes[msgh] + 1
            msg = kind + ": " + msg.split("\n")[0].strip()
            msg = msg if len(msg) <= 70 else msg[:69] + "…"
            logger.error(msg + f" ({count})")

    # Methods that must be overloaded

    def get_pixel_ratio(self):
        """ Get the float ratio between logical and physical pixels.
        """
        raise NotImplementedError()

    def get_logical_size(self):
        """ Get the logical size in float pixels.
        """
        raise NotImplementedError()

    def get_physical_size(self):
        """ Get the physical size in integer pixels.
        """
        raise NotImplementedError()

    def set_logical_size(self, width, height):
        """ Set the window size (in logical pixels).
        """
        raise NotImplementedError()

    def request_draw(self):
        """ Request from the main loop to schedule a new draw event,
        so that the canvas will be updated.
        """
        raise NotImplementedError()

    def close(self):
        """ Close the window.
        """
        raise NotImplementedError()

    def is_closed(self):
        """ Get whether the window is closed.
        """
        raise NotImplementedError()
