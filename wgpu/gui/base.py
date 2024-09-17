import sys
import time
from collections import defaultdict

from ._gui_utils import log_exception


class WgpuCanvasInterface:
    """The minimal interface to be a valid canvas.

    Any object that implements these methods is a canvas that wgpu can work with.
    The object does not even have to derive from this class.

    In most cases it's more convenient to subclass :class:`WgpuCanvasBase <wgpu.gui.WgpuCanvasBase>`.
    """

    def __init__(self, *args, **kwargs):
        # The args/kwargs are there because we may be mixed with e.g. a Qt widget
        super().__init__(*args, **kwargs)
        self._canvas_context = None

    def get_surface_info(self):
        """Get information about the native window / surface.

        This is used to obtain a surface id, so that wgpu can render to the
        region of the screen occupied by the canvas. Should return None for
        offscreen canvases. Otherwise, this should return a dict with a "window"
        field. On Linux the dict should contain more fields, see the existing
        implementations for reference.
        """
        return None

    def get_physical_size(self):
        """Get the physical size of the canvas in integer pixels."""
        raise NotImplementedError()

    def get_context(self, kind="webgpu"):
        """Get the ``GPUCanvasContext`` object corresponding to this canvas.

        The context is used to obtain a texture to render to, and to
        present that texture to the canvas. This class provides a
        default implementation to get the appropriate context.

        The ``kind`` argument is a remnant from the WebGPU spec and
        must always be "webgpu".
        """
        # Note that this function is analog to HtmlCanvas.getContext(), except
        # here the only valid arg is 'webgpu', which is also made the default.
        assert kind == "webgpu"
        if self._canvas_context is None:
            # Get the active wgpu backend module
            backend_module = sys.modules["wgpu"].gpu.__module__
            if backend_module == "wgpu._classes":
                raise RuntimeError(
                    "A backend must be selected (e.g. with request_adapter()) before canvas.get_context() can be called."
                )
            # Instantiate the context
            CC = sys.modules[backend_module].GPUCanvasContext  # noqa: N806
            self._canvas_context = CC(self)
        return self._canvas_context


class WgpuCanvasBase(WgpuCanvasInterface):
    """A convenient base canvas class.

    This class provides a uniform API and implements common
    functionality, to increase consistency and reduce code duplication.
    It is convenient (but not strictly necessary) for canvas classes
    to inherit from this class (but all builtin canvases do).

    This class provides an API for scheduling draws (``request_draw()``)
    and implements a mechanism to call the provided draw function
    (``draw_frame()``) and then present the result to the canvas.

    This class also implements draw rate limiting, which can be set
    with the ``max_fps`` attribute (default 30). For benchmarks you may
    also want to set ``vsync`` to False.
    """

    def __init__(self, *args, max_fps=30, vsync=True, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_draw_time = 0
        self._max_fps = float(max_fps)
        self._vsync = bool(vsync)

    def __del__(self):
        # On delete, we call the custom close method.
        try:
            self.close()
        except Exception:
            pass
        # Since this is sometimes used in a multiple inheritance, the
        # superclass may (or may not) have a __del__ method.
        try:
            super().__del__()
        except Exception:
            pass

    def draw_frame(self):
        """The function that gets called at each draw.

        You can implement this method in a subclass, or set it via a
        call to request_draw().
        """
        pass

    def request_draw(self, draw_function=None):
        """Schedule a new draw event.

        This function does not perform a draw directly, but schedules
        a draw event at a suitable moment in time. In the draw event
        the draw function is called, and the resulting rendered image
        is presented to screen.

        Arguments:
            draw_function (callable or None): The function to set as the new draw
                function. If not given or None, the last set draw function is used.

        """
        if draw_function is not None:
            self.draw_frame = draw_function
        self._request_draw()

    def _draw_frame_and_present(self):
        """Draw the frame and present the result.

        Errors are logged to the "wgpu" logger. Should be called by the
        subclass at an appropriate time.
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

    def set_title(self, title):
        """Set the window title."""
        raise NotImplementedError()

    def close(self):
        """Close the window."""
        pass

    def is_closed(self):
        """Get whether the window is closed."""
        raise NotImplementedError()

    def _request_draw(self):
        """GUI-specific implementation for ``request_draw()``.

        * This should invoke a new draw at a later time.
        * The call itself should return directly.
        * Multiple calls should result in a single new draw.
        * Preferably the ``max_fps`` and ``vsync`` are honored.
        """
        raise NotImplementedError()


class WgpuAutoGui:
    """Mixin class for canvases implementing autogui.

    This class provides a common API for handling events and registering
    event handlers. It adds to :class:`WgpuCanvasBase <wgpu.gui.WgpuCanvasBase>`
    that interactive examples and applications can be written in a
    generic way (no-GUI specific code).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_event_time = 0
        self._pending_events = {}
        self._event_handlers = defaultdict(list)

    def _get_event_wait_time(self):
        """Calculate the time to wait for the next event dispatching.

        Used for rate-limited events.
        """
        rate = 75  # events per second
        now = time.perf_counter()
        target_time = self._last_event_time + 1.0 / rate
        return max(0, target_time - now)

    def _handle_event_rate_limited(
        self, event, call_later_func, match_keys, accum_keys
    ):
        """Alternative `to handle_event()` for events that must be rate-limited.

        If any of the ``match_keys`` keys of the new event differ from the currently
        pending event, the old event is dispatched now. The ``accum_keys`` keys of
        the current and new event are added together (e.g. to accumulate wheel delta).

        The (accumulated) event is handled in the following cases:
        * When the timer runs out.
        * When a non-rate-limited event is dispatched.
        * When a rate-limited event of the same type is scheduled
          that has different match_keys (e.g. modifiers changes).

        Subclasses that use this method must use ``_handle_event_and_flush()``
        where they would otherwise call ``handle_event()``, to preserve event order.
        """
        event_type = event["event_type"]
        event.setdefault("time_stamp", time.perf_counter())
        # We may need to emit the old event. Otherwise, we need to update the new one.
        old = self._pending_events.get(event_type, None)
        if old:
            if any(event[key] != old[key] for key in match_keys):
                self.handle_event(old)
            else:
                for key in accum_keys:
                    event[key] = old[key] + event[key]
        # Make sure that we have scheduled a moment to handle events
        if not self._pending_events:
            call_later_func(self._get_event_wait_time(), self._handle_pending_events)
        # Store the event object
        self._pending_events[event_type] = event

    def _handle_event_and_flush(self, event):
        """Call handle_event after flushing any pending (rate-limited) events."""
        event.setdefault("time_stamp", time.perf_counter())
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
        https://jupyter-rfb.readthedocs.io/en/stable/events.html

        The default implementation dispatches the event to the
        registered event handlers.

        Arguments:
            event (dict): the event to handle.
        """
        # Collect callbacks
        event_type = event.get("event_type")
        callbacks = self._event_handlers[event_type] + self._event_handlers["*"]
        # Dispatch
        for _, callback in callbacks:
            with log_exception(f"Error during handling {event['event_type']} event"):
                if event.get("stop_propagation", False):
                    break
                callback(event)

    def add_event_handler(self, *args, order=0):
        """Register an event handler to receive events.

        Arguments:
            callback (callable): The event handler. Must accept a single event argument.
            *types (list of strings): A list of event types.
            order (int): The order in which the handler is called. Lower numbers are called first. Default is 0.

        For the available events, see
        https://jupyter-rfb.readthedocs.io/en/stable/events.html.

        The callback is stored, so it can be a lambda or closure. This also
        means that if a method is given, a reference to the object is held,
        which may cause circular references or prevent the Python GC from
        destroying that object.

        Example:

        .. code-block:: py

            def my_handler(event):
                print(event)

            canvas.add_event_handler(my_handler, "pointer_up", "pointer_down")

        Can also be used as a decorator:

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
                self._event_handlers[type].append((order, _callback))
                self._event_handlers[type].sort(key=lambda x: x[0])
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
            self._event_handlers[type] = [
                (o, cb) for o, cb in self._event_handlers[type] if cb is not callback
            ]
