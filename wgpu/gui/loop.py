from collections import defaultdict


class HasEvents:
    """An object derived from this class can register events."""

    # todo: make the types of events appear in the docs
    # todo: use to check subscribing to events
    _event_types = ()

    def __init__(self):

        self._pending_events = {}
        self._event_handlers = defaultdict(set)

    def add_event_handler(self, *args):
        """Register an event handler to receive events.

        Arguments:
            callback (callable): The event handler. Must accept a single event argument.
            *types (list of strings): A list of event types.

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


class BaseLoop(HasEvents):
    """Base loop implementation."""

    _event_types = "animate", "before_render", "render"

    def run(self):
        raise NotImplementedError()


class WgpuCanvasWithEvents:

    _event_types = "pointer_down", "pointer_up", ...

    def __init__(self, loop):
        self._loop = loop

    @property
    def loop(self):
        return self._loop


class BaseLoopCanvasAdapter:
    pass


class BaseLoopRunnerAdapter:
    pass
