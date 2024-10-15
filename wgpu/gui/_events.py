import time
from collections import defaultdict, deque

from ._gui_utils import log_exception
from ..enums import Enum


class WgpuEventType(Enum):
    """The WgpuEventType enum specifies the possible events for a WgpuCanvas.

    This includes the events from the jupyter_rfb event spec (see
    https://jupyter-rfb.readthedocs.io/en/stable/events.html) plus some
    wgpu-specific events.
    """

    # Jupter_rfb spec

    resize = None  #: The canvas has changed size. Has 'width' and 'height' in logical pixels, 'pixel_ratio'.
    close = None  #: The canvas is closed. No additional fields.
    pointer_down = None  #: The pointing device is pressed down. Has 'x', 'y', 'button', 'butons', 'modifiers', 'ntouches', 'touches'.
    pointer_up = None  #: The pointing device is released. Same fields as pointer_down.
    pointer_move = None  #: The  pointing device is moved. Same fields as pointer_down.
    double_click = None  #: A double-click / long-tap. This event looks like a pointer event, but without the touches.
    wheel = None  #: The mouse-wheel is used (scrolling), or the touchpad/touchscreen is scrolled/pinched. Has 'dx', 'dy', 'x', 'y', 'modifiers'.
    key_down = None  #: A key is pressed down. Has 'key', 'modifiers'.
    key_up = None  #: A key is released. Has 'key', 'modifiers'.

    # Our extra events

    before_draw = (
        None  #: Event emitted right before a draw is performed. Has no extra fields.
    )
    animate = None  #: Animation event. Has 'step' representing the step size in seconds. This is stable, except when the 'catch_up' field is nonzero.


class EventEmitter:
    """The EventEmitter stores event handlers, collects incoming events, and dispatched them.

    Subsequent events of ``event_type`` 'pointer_move' and 'wheel' are merged.
    """

    _EVENTS_THAT_MERGE = {
        "pointer_move": {
            "match_keys": {"buttons", "modifiers", "ntouches"},
            "accum_keys": {},
        },
        "wheel": {
            "match_keys": {"modifiers"},
            "accum_keys": {"dx", "dy"},
        },
    }

    def __init__(self):
        self._pending_events = deque()
        self._event_handlers = defaultdict(list)

    def add_handler(self, *args, order=0):
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
            if not (type == "*" or type in WgpuEventType):
                raise ValueError(f"Adding handler with invalid event_type: '{type}'")

        def decorator(_callback):
            for type in types:
                self._event_handlers[type].append((order, _callback))
                self._event_handlers[type].sort(key=lambda x: x[0])
            return _callback

        if decorating:
            return decorator
        return decorator(callback)

    def remove_handler(self, callback, *types):
        """Unregister an event handler.

        Arguments:
            callback (callable): The event handler.
            *types (list of strings): A list of event types.
        """
        for type in types:
            self._event_handlers[type] = [
                (o, cb) for o, cb in self._event_handlers[type] if cb is not callback
            ]

    def submit(self, event):
        """Submit an event.

        Events are emitted later by the scheduler.
        """
        event_type = event["event_type"]
        if event_type not in WgpuEventType:
            raise ValueError(f"Submitting with invalid event_type: '{event_type}'")

        event.setdefault("time_stamp", time.perf_counter())
        event_merge_info = self._EVENTS_THAT_MERGE.get(event_type, None)

        if event_merge_info and self._pending_events:
            # Try merging the event with the last one
            last_event = self._pending_events[-1]
            if last_event["event_type"] == event_type:
                match_keys = event_merge_info["match_keys"]
                accum_keys = event_merge_info["accum_keys"]
                if any(event[key] != last_event[key] for key in match_keys):
                    # Keys don't match: new event
                    self._pending_events.append(event)
                else:
                    # Update last event (i.e. merge)
                    for key in accum_keys:
                        last_event[key] += event[key]
        else:
            self._pending_events.append(event)

    def flush(self):
        """Dispatch all pending events.

        This should generally be left to the scheduler.
        """
        while True:
            try:
                event = self._pending_events.popleft()
            except IndexError:
                break
            # Collect callbacks
            event_type = event.get("event_type")
            callbacks = self._event_handlers[event_type] + self._event_handlers["*"]
            # Dispatch
            for _order, callback in callbacks:
                if event.get("stop_propagation", False):
                    break
                with log_exception(f"Error during handling {event_type} event"):
                    callback(event)
