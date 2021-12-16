import os
import sys
import time
import logging
import ctypes.util

logger = logging.getLogger("wgpu")


class WgpuCanvasInterface:
    """This is the interface that a canvas object must implement in order
    to be a valid canvas that wgpu can work with.
    """

    # NOTE: This is an interface - it should not be necessary to actually subclass

    def __init__(self, *args, **kwargs):
        # The args/kwargs are there because we may be mixed with e.g. a Qt widget
        super().__init__(*args, **kwargs)
        self._canvas_context = None

    def get_window_id(self):
        """Get the native window id. This is used to obtain a surface id,
        so that wgpu can render to the region of the screen occupied by the canvas.
        """
        raise NotImplementedError()

    def get_display_id(self):
        """Get the native display id on Linux. This is needed in addition to the
        window id to obtain a surface id. The default implementation calls into
        the X11 lib to get the display id.
        """
        # Re-use to avoid creating loads of id's
        if getattr(self, "_display_id", None) is not None:
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
        """Get the physical size of the canvas in integer pixels."""
        raise NotImplementedError()

    def get_context(self, kind="gpupresent"):
        """Get the GPUCanvasContext object corresponding to this canvas,
        which can be used to e.g. obtain a texture to render to.
        """
        # Note that this function is analog to HtmlCanvas.get_context(), except
        # here the only valid arg is 'gpupresent', which is also made the default.
        assert kind == "gpupresent"
        if self._canvas_context is None:
            # Get the active wgpu backend module
            backend_module = sys.modules["wgpu"].GPU.__module__
            # Instantiate the context
            PC = sys.modules[backend_module].GPUCanvasContext  # noqa: N806
            self._canvas_context = PC(self)
        return self._canvas_context


class WgpuCanvasBase(WgpuCanvasInterface):
    """An abstract class extending :class:`WgpuCanvasInterface`,
    that provides a base canvas for various GUI toolkits, so
    that basic canvas functionality is available via a common API.

    It is convenient - but not required - to use this class (or any of its
    subclasses) to use wgpu-py.
    """

    def __init__(self, *args, max_fps=30, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_draw_time = 0
        self._max_fps = float(max_fps)
        self._err_hashes = {}

    def draw_frame(self):
        """The function that gets called at each draw. You can implement
        this method in a subclass, or set it via a call to request_draw().
        """
        pass

    def request_draw(self, draw_function=None):
        """Request from the main loop to schedule a new draw event,
        so that the canvas will be updated. If draw_function is not
        given, the last set drawing function is used.
        """
        if draw_function is not None:
            self.draw_frame = draw_function
        self._request_draw()

    def _draw_frame_and_present(self):
        """Draw the frame and present the result. Errors are logged to the
        "wgpu" logger. Should be called by the subclass at an appropriate time.
        """
        self._last_draw_time = time.perf_counter()
        # Perform the user-defined drawing code. When this errors,
        # we should report the error and then continue, otherwise we crash.
        # Returns the result of the context's present() call or None.
        try:
            self.draw_frame()
        except Exception as err:
            self._log_exception("Draw error", err)
        try:
            if self._canvas_context:
                return self._canvas_context.present()
        except Exception as err:
            self._log_exception("Present error", err)

    def _get_draw_wait_time(self):
        """Get time (in seconds) to wait until the next draw in order to honour max_fps."""
        now = time.perf_counter()
        target_time = self._last_draw_time + 1.0 / self._max_fps
        return max(0, target_time - now)

    def _log_exception(self, kind, err):
        """Log the given exception instance, but only log a one-liner for
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
        """Get the float ratio between logical and physical pixels."""
        raise NotImplementedError()

    def get_logical_size(self):
        """Get the logical size in float pixels."""
        raise NotImplementedError()

    def get_physical_size(self):
        """Get the physical size in integer pixels."""
        raise NotImplementedError()

    def set_logical_size(self, width, height):
        """Set the window size (in logical pixels)."""
        raise NotImplementedError()

    def close(self):
        """Close the window."""
        raise NotImplementedError()

    def is_closed(self):
        """Get whether the window is closed."""
        raise NotImplementedError()

    def _request_draw(self):
        """This should invoke a new draw in a later event loop
        iteration (i.e. the call itself should return directly).
        Multiple calls should result in a single new draw. Preferably
        the FPS is limited to avoid draining CPU and power.
        """
        raise NotImplementedError()
