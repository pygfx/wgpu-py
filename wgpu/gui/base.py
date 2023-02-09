import os
import sys
import time
import weakref
import logging
from contextlib import contextmanager
import ctypes.util
from collections import defaultdict


logger = logging.getLogger("wgpu")

err_hashes = {}


@contextmanager
def log_exception(kind):
    """Context manager to log any exceptions, but only log a one-liner
    for subsequent occurances of the same error to avoid spamming by
    repeating errors in e.g. a draw function or event callback.
    """
    try:
        yield
    except Exception as err:
        # Store exc info for postmortem debugging
        exc_info = list(sys.exc_info())
        exc_info[2] = exc_info[2].tb_next  # skip *this* function
        sys.last_type, sys.last_value, sys.last_traceback = exc_info
        # Show traceback, or a one-line summary
        msg = str(err)
        msgh = hash(msg)
        if msgh not in err_hashes:
            # Provide the exception, so the default logger prints a stacktrace.
            # IDE's can get the exception from the root logger for PM debugging.
            err_hashes[msgh] = 1
            logger.error(kind, exc_info=err)
        else:
            # We've seen this message before, return a one-liner instead.
            err_hashes[msgh] = count = err_hashes[msgh] + 1
            msg = kind + ": " + msg.split("\n")[0].strip()
            msg = msg if len(msg) <= 70 else msg[:69] + "…"
            logger.error(msg + f" ({count})")


def weakbind(method):
    """Replace a bound method with a callable object that stores the `self` using a weakref."""
    ref = weakref.ref(method.__self__)
    class_func = method.__func__
    del method

    def proxy(*args, **kwargs):
        self = ref()
        if self is not None:
            return class_func(self, *args, **kwargs)

    proxy.__name__ = class_func.__name__
    return proxy


class WgpuCanvasInterface:
    """The minimal interface to be a valid canvas.

    Any object that implements these methods is a canvas that wgpu can work with.
    The object does not even have to derive from this class.

    In most cases it's more convenient to subclass `gui.WgpuCanvasBase`.
    """

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
    """A canvas class that provides a basis for all GUI toolkits.

    This class implements common functionality, to realize a common API
    and avoid code duplication. It is convenient (but not strictly necessary)
    for canvas classes to inherit from this class (all builtin canvases do).

    Amongst other things, this class implements draw rate limiting,
    which can be set with the ``max_fps`` attribute (default 30). For
    benchmarks you may also want to set ``vsync`` to False.
    """

    def __init__(self, *args, max_fps=30, vsync=True, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_draw_time = 0
        self._max_fps = float(max_fps)
        self._vsync = bool(vsync)
        self._err_hashes = {}

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

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
        with log_exception("Draw error"):
            self.draw_frame()
        with log_exception("Present error"):
            if self._canvas_context:
                return self._canvas_context.present()

    def _get_draw_wait_time(self):
        """Get time (in seconds) to wait until the next draw in order to honour max_fps."""
        now = time.perf_counter()
        target_time = self._last_draw_time + 1.0 / self._max_fps
        return max(0, target_time - now)

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
        pass

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


class WgpuAutoGui:
    """Mixin class for canvases implementing autogui.

    AutoGui canvases provide an API for handling events and registering event
    handlers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_event_time = 0
        self._pending_events = {}
        self._event_handlers = defaultdict(set)

    def _get_event_wait_time(self):
        """Calculate the time to wait for the next event dispatching
        (for rate-limited events)."""
        rate = 75  # events per second
        now = time.perf_counter()
        target_time = self._last_event_time + 1.0 / rate
        return max(0, target_time - now)

    def _handle_event_rate_limited(self, ev, call_later_func, match_keys, accum_keys):
        """Alternative `to handle_event()` for events that must be rate-limted.
        If any of the `match_keys` keys of the new event differ from the currently
        pending event, the old event is dispatched now. The `accum_keys` keys of
        the current and new event are added together (e.g. to accumulate wheel delta).

        The (accumulated) event is handled in the following cases:
        * When the timer runs out.
        * When a non-rate-limited event is dispatched.
        * When a rate-limited event of the same type is scheduled
          that has different match_keys (e.g. modifiers changes).

        Subclasses that use this method must use ``_handle_event_and_flush()``
        where they would otherwise call ``handle_event()``, to preserve event order.
        """
        event_type = ev["event_type"]
        # We may need to emit the old event. Otherwise, we need to update the new one.
        old = self._pending_events.get(event_type, None)
        if old:
            if any(ev[key] != old[key] for key in match_keys):
                self.handle_event(old)
            else:
                for key in accum_keys:
                    ev[key] = old[key] + ev[key]
        # Make sure that we have scheduled a moment to handle events
        if not self._pending_events:
            call_later_func(self._get_event_wait_time(), self._handle_pending_events)
        # Store the event object
        self._pending_events[event_type] = ev

    def _handle_event_and_flush(self, event):
        """Call handle_event after flushing any pending (rate-limited) events."""
        self._handle_pending_events()
        self.handle_event(event)

    def _handle_pending_events(self):
        """Handle any pending rate-limited events."""
        if self._pending_events:
            events = self._pending_events.values()
            self._last_event_time = time.perf_counter()
            self._pending_events = {}
            for ev in events:
                self.handle_event(ev)

    def handle_event(self, event):
        """Handle an incoming event.

        Subclasses can overload this method. Events include widget
        resize, mouse/touch interaction, key events, and more. An event
        is a dict with at least the key event_type. For details, see
        https://jupyter-rfb.readthedocs.io/en/latest/events.html

        The default implementation dispatches the event to the
        registered event handlers.
        """
        # Collect callbacks
        event_type = event.get("event_type")
        callbacks = self._event_handlers[event_type] | self._event_handlers["*"]
        # Dispatch
        for callback in callbacks:
            with log_exception(f"Error during handling {event['event_type']} event"):
                callback(event)

    def add_event_handler(self, *args):
        """Register an event handler.

        Arguments:
            callback (callable): The event handler. Must accept a
                single event argument.
            *types (list of strings): A list of event types.

        For the available events, see
        https://jupyter-rfb.readthedocs.io/en/latest/events.html

        Can also be used as a decorator.

        Example:

        .. code-block:: py

            def my_handler(event):
                print(event)

            canvas.add_event_handler(my_handler, "pointer_up", "pointer_down")

        Decorator usage example:

        .. code-block:: py

            @canvas.add_event_handler("pointer_up", "pointer_down")
            def my_handler(event):
                print(event)

        Catch 'm all:

        .. code-block:: py

            canvas.add_event_handler(my_handler, "*")

        """
        decorating = not callable(args[0])
        callback = None if decorating else args[0]
        types = args if decorating else args[1:]

        if not types:
            raise TypeError("No event types are given to add_event_handler.")
        for type in types:
            if not isinstance(type, str):
                raise TypeError(f"Event types must be str, but got {type}")

        def decorator(_callback):
            for type in types:
                self._event_handlers[type].add(_callback)
            return _callback

        if decorating:
            return decorator
        return decorator(callback)

    def remove_event_handler(self, callback, *types):
        """Unregister an event handler.

        Arguments:
            callback (callable): The event handler.
            *types (list of strings): A list of event types.
        """
        for type in types:
            self._event_handlers[type].remove(callback)
