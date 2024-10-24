"""
Test the EventEmitter.
"""

import time

from wgpu.gui._events import EventEmitter, WgpuEventType
from testutils import run_tests
import pytest


def test_events_event_types():
    ee = EventEmitter()

    def handler(event):
        pass

    # All these are valid
    valid_types = list(WgpuEventType)
    ee.add_handler(handler, *valid_types)

    # This is not
    with pytest.raises(ValueError):
        ee.add_handler(handler, "not_a_valid_event_type")

    # This is why we use key  events below :)


def test_events_basic():
    ee = EventEmitter()

    values = []

    def handler(event):
        values.append(event["value"])

    ee.add_handler(handler, "key_down")

    ee.submit({"event_type": "key_down", "value": 1})
    ee.submit({"event_type": "key_down", "value": 2})
    assert values == []

    ee.flush()
    ee.submit({"event_type": "key_down", "value": 3})
    assert values == [1, 2]

    ee.flush()
    assert values == [1, 2, 3]

    # Removing a handler affects all events since the last flush
    ee.submit({"event_type": "key_down", "value": 4})
    ee.remove_handler(handler, "key_down")
    ee.submit({"event_type": "key_down", "value": 5})
    ee.flush()
    assert values == [1, 2, 3]


def test_events_handler_arg_position():
    ee = EventEmitter()

    def handler(event):
        pass

    with pytest.raises(TypeError):
        ee.add_handler("key_down", "key_up", handler)

    with pytest.raises(TypeError):
        ee.add_handler("key_down", handler, "key_up")


def test_events_handler_decorated():
    ee = EventEmitter()

    values = []

    @ee.add_handler("key_down", "key_up")
    def handler(event):
        values.append(event["value"])

    ee.submit({"event_type": "key_down", "value": 1})
    ee.submit({"event_type": "key_up", "value": 2})
    ee.flush()
    assert values == [1, 2]


def test_direct_emit_():
    ee = EventEmitter()

    values = []

    @ee.add_handler("key_down", "key_up")
    def handler(event):
        values.append(event["value"])

    ee.submit({"event_type": "key_down", "value": 1})
    ee.flush()
    ee.submit({"event_type": "key_up", "value": 2})
    ee.emit({"event_type": "key_up", "value": 3})  # goes before pending events
    ee.submit({"event_type": "key_up", "value": 4})
    ee.flush()
    ee.submit({"event_type": "key_up", "value": 5})

    assert values == [1, 3, 2, 4]


def test_events_two_types():
    ee = EventEmitter()

    values = []

    def handler(event):
        values.append(event["value"])

    ee.add_handler(handler, "key_down", "key_up")

    ee.submit({"event_type": "key_down", "value": 1})
    ee.submit({"event_type": "key_up", "value": 2})
    ee.flush()
    assert values == [1, 2]

    ee.remove_handler(handler, "key_down")
    ee.submit({"event_type": "key_down", "value": 3})
    ee.submit({"event_type": "key_up", "value": 4})
    ee.flush()
    assert values == [1, 2, 4]

    ee.remove_handler(handler, "key_up")
    ee.submit({"event_type": "key_down", "value": 5})
    ee.submit({"event_type": "key_up", "value": 6})
    ee.flush()
    assert values == [1, 2, 4]


def test_events_two_handlers():
    ee = EventEmitter()

    values = []

    def handler1(event):
        values.append(100 + event["value"])

    def handler2(event):
        values.append(200 + event["value"])

    ee.add_handler(handler1, "key_down")
    ee.add_handler(handler2, "key_down")

    ee.submit({"event_type": "key_down", "value": 1})
    ee.flush()
    assert values == [101, 201]

    ee.remove_handler(handler1, "key_down")
    ee.submit({"event_type": "key_down", "value": 2})
    ee.flush()
    assert values == [101, 201, 202]

    ee.remove_handler(handler2, "key_down")
    ee.submit({"event_type": "key_down", "value": 3})
    ee.flush()
    assert values == [101, 201, 202]


def test_events_handler_order():
    ee = EventEmitter()

    values = []

    def handler1(event):
        values.append(100 + event["value"])

    def handler2(event):
        values.append(200 + event["value"])

    def handler3(event):
        values.append(300 + event["value"])

    # Handlers are called in the order they were added.
    # This is what most systems use. Except Vispy (and therefore Napari),
    # which causes them a lot of trouble:
    # https://github.com/vispy/vispy/blob/af84742/vispy/util/event.py#L263-L264
    # https://github.com/napari/napari/pull/7150
    # https://github.com/napari/napari-animation/pull/234
    ee.add_handler(handler1, "key_down")
    ee.add_handler(handler2, "key_down")
    ee.add_handler(handler3, "key_down")

    ee.submit({"event_type": "key_down", "value": 1})
    ee.flush()
    assert values == [101, 201, 301]

    # Now re-connect with priorities
    values.clear()
    ee.add_handler(handler1, "key_down", order=0)  # default
    ee.add_handler(handler2, "key_down", order=2)
    ee.add_handler(handler3, "key_down", order=1)

    ee.submit({"event_type": "key_down", "value": 1})
    ee.flush()
    assert values == [101, 301, 201]

    # Another run using negative priorities too
    values.clear()
    ee.add_handler(handler1, "key_down", order=1)  # default
    ee.add_handler(handler2, "key_down", order=-2)
    ee.add_handler(handler3, "key_down", order=-1)

    ee.submit({"event_type": "key_down", "value": 1})
    ee.flush()
    assert values == [201, 301, 101]

    # Use floats!
    values.clear()
    ee.add_handler(handler1, "key_down", order=0.33)  # default
    ee.add_handler(handler2, "key_down", order=0.22)
    ee.add_handler(handler3, "key_down", order=0.11)

    ee.submit({"event_type": "key_down", "value": 1})
    ee.flush()
    assert values == [301, 201, 101]


def test_events_duplicate_handler():
    ee = EventEmitter()

    values = []

    def handler(event):
        values.append(event["value"])

    # Registering for the same event_type twice just adds it once
    ee.add_handler(handler, "key_down")
    ee.add_handler(handler, "key_down")

    ee.submit({"event_type": "key_down", "value": 1})
    ee.flush()
    assert values == [1]

    ee.remove_handler(handler, "key_down")
    ee.submit({"event_type": "key_down", "value": 2})
    ee.flush()
    assert values == [1]


def test_events_duplicate_handler_with_lambda():
    ee = EventEmitter()

    values = []

    def handler(event):
        values.append(event["value"])

    # Cannot discern now, these are two different handlers
    ee.add_handler(lambda e: handler(e), "key_down")
    ee.add_handler(lambda e: handler(e), "key_down")

    ee.submit({"event_type": "key_down", "value": 1})
    ee.flush()
    assert values == [1, 1]

    ee.remove_handler(handler, "key_down")
    ee.submit({"event_type": "key_down", "value": 2})
    ee.flush()
    assert values == [1, 1, 2, 2]


def test_merging_events():
    ee = EventEmitter()

    events = []

    @ee.add_handler("resize", "wheel", "pointer_move", "key_down")
    def handler(event):
        events.append(event)

    ee.submit({"event_type": "resize", "width": 100})
    ee.submit({"event_type": "resize", "width": 102})
    ee.submit({"event_type": "resize", "width": 104})

    ee.submit({"event_type": "wheel", "dx": 1, "dy": 0})
    ee.submit({"event_type": "wheel", "dx": 1, "dy": 0})
    ee.submit({"event_type": "wheel", "dx": 3, "dy": 0})

    ee.submit({"event_type": "pointer_move", "x": 120, "modifiers": ()})
    ee.submit({"event_type": "pointer_move", "x": 122, "modifiers": ()})
    ee.submit({"event_type": "pointer_move", "x": 123, "modifiers": ()})

    ee.submit({"event_type": "pointer_move", "x": 125, "modifiers": ("Ctrl")})

    ee.submit({"event_type": "resize", "width": 106})
    ee.submit({"event_type": "resize", "width": 108})

    ee.submit({"event_type": "key_down", "value": 1})
    ee.submit({"event_type": "key_down", "value": 2})

    ee.flush()

    assert len(events) == 7

    # First three event types are merges
    assert events[0]["width"] == 104
    assert events[1]["dx"] == 5
    assert events[2]["x"] == 123

    # Next one is separate because of different match_keys
    assert events[3]["x"] == 125

    # The second series of resize events are separate because they are
    # not consecutive with the previous series
    assert events[4]["width"] == 108

    # Key events are not merged
    assert events[5]["value"] == 1
    assert events[6]["value"] == 2


def test_mini_benchmark():
    # Can be used to tweak internals of the EventEmitter and see the
    # effect on performance.

    ee = EventEmitter()

    def handler(event):
        pass

    t0 = time.perf_counter()
    for _ in range(1000):
        ee.add_handler(lambda e: handler(e), "key_down", order=1)
        ee.add_handler(lambda e: handler(e), "key_down", order=2)
    t1 = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(100):
        ee.submit({"event_type": "key_down", "value": 2})
        ee.flush()
    t2 = time.perf_counter() - t0

    print(f"add_handler: {1000*t1:0.0f} ms, emit: {1000*t2:0.0f} ms")


if __name__ == "__main__":
    run_tests(globals())
