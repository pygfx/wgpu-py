import wgpu
from wgpu.gui.events import Event, EventDispatcher, EventTarget, PointerEvent


class Node(EventTarget):
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent


def test_event_target_mixin():
    c = EventTarget()

    # It's a mixin
    assert not isinstance(c, wgpu.gui.WgpuCanvasBase)

    # It's event handling mechanism should be fully functional
    events = []

    def handler(event):
        events.append(event["value"])

    c.add_event_handler(handler, "foo", "bar")
    c.handle_event(Event(type="foo", value=1))
    c.handle_event(Event(type="bar", value=2))
    c.handle_event(Event(type="spam", value=3))
    c.remove_event_handler(handler, "foo")
    c.handle_event(Event(type="foo", value=4))
    c.handle_event(Event(type="bar", value=5))
    c.handle_event(Event(type="spam", value=6))
    c.remove_event_handler(handler, "bar")
    c.handle_event(Event(type="foo", value=7))
    c.handle_event(Event(type="bar", value=8))
    c.handle_event(Event(type="spam", value=9))

    assert events == [1, 2, 5]


def test_event_bubbling():
    """
    Check that events bubble up in hierarchy
    """
    dispatcher_called = 0
    root_called = 0
    child_called = 0

    def dispatcher_callback(event):
        nonlocal dispatcher_called
        dispatcher_called += 1

    def root_callback(event):
        nonlocal root_called
        root_called += 1

    def child_callback(event):
        nonlocal child_called
        child_called += 1

    dispatcher = EventDispatcher()
    root = Node()
    child = Node(parent=root)

    dispatcher.add_event_handler(dispatcher_callback, "foo")
    root.add_event_handler(root_callback, "foo")
    child.add_event_handler(child_callback, "foo")

    event = Event(type="foo", target=child)
    dispatcher.handle_event(event)

    assert child_called == 1
    assert root_called == 1
    assert dispatcher_called == 1

    # Make dispatcher part of the bubble tree
    root.parent = dispatcher
    event = Event(type="foo", target=child)
    dispatcher.handle_event(event)

    assert child_called == 2
    assert root_called == 2
    assert dispatcher_called == 2


def test_event_stop_propagation():
    """
    Check that bubbling stops when stop_propagation
    is called on an event.
    """
    root_called = 0
    child_called = 0

    def root_callback(event):
        nonlocal root_called
        root_called += 1

    def child_callback(event):
        nonlocal child_called
        child_called += 1

    def child_prevent_callback(event):
        nonlocal child_called
        # Prevent bubbling up
        event.stop_propagation()
        child_called += 1

    root = Node()
    child = Node(parent=root)

    dispatcher = EventDispatcher()
    root.add_event_handler(root_callback, "foo")
    child.add_event_handler(child_callback, "foo")

    event = Event(type="foo", target=child)
    dispatcher.handle_event(event)

    assert child_called == 1
    assert root_called == 1

    child.remove_event_handler(child_callback, "foo")
    child.add_event_handler(child_prevent_callback, "foo")

    event = Event(type="foo", target=child)
    dispatcher.handle_event(event)

    assert child_called == 2
    assert root_called == 1


def test_event_bubbles_attribute():
    root_called = 0
    child_called = 0

    def root_callback(event):
        nonlocal root_called
        root_called += 1

    def child_callback(event):
        nonlocal child_called
        child_called += 1

    root = Node()
    child = Node(parent=root)
    root.add_event_handler(root_callback, "foo")
    child.add_event_handler(child_callback, "foo")

    dispatcher = EventDispatcher()

    event = Event(type="foo", target=child, bubbles=True)
    dispatcher.handle_event(event)

    assert child_called == 1
    assert root_called == 1

    event = Event(type="foo", target=child, bubbles=False)
    dispatcher.handle_event(event)

    assert child_called == 2
    assert root_called == 1


def test_pointer_event_capture():
    root_called = 0
    child_called = 0

    def root_callback(event):
        nonlocal root_called
        root_called += 1

    def child_callback(event):
        nonlocal child_called
        child_called += 1

    root = Node()
    child = Node(parent=root)
    root.add_event_handler(root_callback, "pointer_down")
    root.add_event_handler(root_callback, "pointer_move")
    root.add_event_handler(root_callback, "pointer_up")
    child.add_event_handler(child_callback, "pointer_down")
    child.add_event_handler(child_callback, "pointer_move")
    child.add_event_handler(child_callback, "pointer_up")

    dispatcher = EventDispatcher()
    dispatcher.handle_event(
        PointerEvent(
            type="pointer_down", x=0, y=0, button=1, pointer_id=0, target=child
        )
    )
    dispatcher.handle_event(
        PointerEvent(
            type="pointer_move", x=1, y=1, button=1, pointer_id=0, target=child
        )
    )
    dispatcher.handle_event(
        PointerEvent(type="pointer_up", x=1, y=1, button=1, pointer_id=0, target=child)
    )

    assert child_called == 3
    assert root_called == 3

    child.add_event_handler(
        lambda e: child.set_pointer_capture(e.pointer_id), "pointer_down"
    )
    child.add_event_handler(
        lambda e: child.release_pointer_capture(e.pointer_id), "pointer_up"
    )

    dispatcher.handle_event(
        PointerEvent(
            type="pointer_down", x=0, y=0, button=1, pointer_id=1, target=child
        )
    )
    dispatcher.handle_event(
        PointerEvent(
            type="pointer_move", x=1, y=1, button=1, pointer_id=1, target=child
        )
    )
    dispatcher.handle_event(
        PointerEvent(type="pointer_up", x=1, y=1, button=1, pointer_id=1, target=child)
    )

    assert child_called == 6
    assert root_called == 3
