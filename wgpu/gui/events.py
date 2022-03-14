from collections import defaultdict
from time import perf_counter_ns


class Event:
    def __init__(
        self,
        type: str,
        *,
        bubbles=True,
        cancelable=True,
        target: "EventTarget" = None,
        **kwargs,
    ):
        self._type = type
        self._time_stamp = perf_counter_ns() * 1000000
        self._default_prevented = False
        self._bubbles = bubbles
        self._cancelable = cancelable
        self._target = target
        self._data = kwargs

    @property
    def type(self) -> str:
        """A string representing the name of the event."""
        return self._type

    @property
    def time_stamp(self) -> float:
        """The time at which the event was created (in milliseconds)."""
        return self._time_stamp

    @property
    def bubbles(self) -> bool:
        """A boolean value indicating whether or not the event bubbles up through
        the scene tree."""
        return self._bubbles

    @property
    def cancelable(self) -> bool:
        """A boolean value indicating whether the event is cancelable."""
        return self._cancelable

    @property
    def default_prevented(self) -> bool:
        """Indicates whether or not the call to ``prevent_default()`` canceled the
        event."""
        return self._default_prevented

    @property
    def target(self) -> "EventTarget":
        """The target property of the Event interface is a reference to the object
        onto which the event was dispatched."""
        return self._target

    def stop_propagation(self):
        """Stops the propagation of events further along in the scene tree."""
        self._bubbles = False

    def prevent_default(self):
        """Cancels the event (if it is cancelable)."""
        if self._cancelable:
            self._default_prevented = True

    def __getitem__(self, key):
        """Make event work like a dict as well to be compatible with the jupyter_rfb
        event spec."""
        if key == "event_type":
            return self.type
        return getattr(self, key, self._data.get(key))

    def __repr__(self):
        prefix = f"<{type(self).__name__} '{self.type}' "
        data = [
            f"{key}={self[key]}"
            for key in dir(self)
            if not key.startswith("_")
            and key
            not in [
                "bubbles",
                "cancelable",
                "default_prevented",
                "prevent_default",
                "stop_propagation",
                "time_stamp",
                "type",
            ]
        ]
        middle = ", ".join(data)
        suffix = ">"
        return "".join([prefix, middle, suffix])


class KeyboardEvent(Event):
    def __init__(self, *args, key, modifiers=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.key = key
        self.modifiers = modifiers or []


class PointerEvent(Event):
    def __init__(
        self,
        *args,
        x,
        y,
        button=0,
        buttons=None,
        modifiers=None,
        ntouches=0,
        touches=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.x = x
        self.y = y
        self.button = button
        self.buttons = buttons or []
        self.modifiers = modifiers or []
        self.ntouches = ntouches
        self.touches = touches or {}


class WheelEvent(Event):
    def __init__(self, *args, dx, dy, x, y, modifiers=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dx = dx
        self.dy = dy
        self.x = x
        self.y = y
        self.modifiers = modifiers or []


class WindowEvent(Event):
    def __init__(self, *args, width=None, height=None, pixel_ratio=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.width = width
        self.height = height
        self.pixel_ratio = pixel_ratio


class EventTarget:
    """Mixin class for canvases implementing autogui."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._event_handlers = defaultdict(set)

    def handle_event(self, event: Event):
        """Handle an incoming event.

        Subclasses can overload this method. Events include widget
        resize, mouse/touch interaction, key events, and more. An event
        is a dict with at least the key event_type. For details, see
        https://jupyter-rfb.readthedocs.io/en/latest/events.html
        """
        event_type = event.type
        for callback in self._event_handlers[event_type]:
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
        """
        decorating = not callable(args[0])
        callback = None if decorating else args[0]
        types = args if decorating else args[1:]

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
