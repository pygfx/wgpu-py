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

    # This is why we use resize and close events below :)


def test_events_basic():

    ee = EventEmitter()

    values = []
    def handler(event):
        values.append(event["value"])

    ee.add_handler(handler, "resize")

    ee.submit({"event_type": "resize", "value": 1})
    ee.submit({"event_type": "resize", "value": 2})
    assert values == []

    ee.flush()
    ee.submit({"event_type": "resize", "value": 3})
    assert values == [1, 2]

    ee.flush()
    assert values == [1, 2, 3]

    # Removing a handler affects all events since the last flush
    ee.submit({"event_type": "resize", "value": 4})
    ee.remove_handler(handler, "resize")
    ee.submit({"event_type": "resize", "value": 5})
    ee.flush()
    assert values == [1, 2, 3]


def test_events_handler_arg_position():

    ee = EventEmitter()

    def handler(event):
        pass

    with pytest.raises(TypeError):
        ee.add_handler("resize", "close", handler)

    with pytest.raises(TypeError):
        ee.add_handler("resize", handler, "close")


def test_events_handler_decorated():

    ee = EventEmitter()

    values = []

    @ee.add_handler("resize", "close")
    def handler(event):
        values.append(event["value"])

    ee.submit({"event_type": "resize", "value": 1})
    ee.submit({"event_type": "close", "value": 2})
    ee.flush()
    assert values == [1, 2]


def test_events_two_types():

    ee = EventEmitter()

    values = []
    def handler(event):
        values.append(event["value"])

    ee.add_handler(handler, "resize", "close")

    ee.submit({"event_type": "resize", "value": 1})
    ee.submit({"event_type": "close", "value": 2})
    ee.flush()
    assert values == [1, 2]

    ee.remove_handler(handler, "resize")
    ee.submit({"event_type": "resize", "value": 3})
    ee.submit({"event_type": "close", "value": 4})
    ee.flush()
    assert values == [1, 2, 4]

    ee.remove_handler(handler, "close")
    ee.submit({"event_type": "resize", "value": 5})
    ee.submit({"event_type": "close", "value": 6})
    ee.flush()
    assert values == [1, 2, 4]


def test_events_two_handlers():

    ee = EventEmitter()

    values = []

    def handler1(event):
        values.append(100 + event["value"])

    def handler2(event):
        values.append(200 + event["value"])

    ee.add_handler(handler1, "resize")
    ee.add_handler(handler2, "resize")

    ee.submit({"event_type": "resize", "value": 1})
    ee.flush()
    assert values == [101, 201]

    ee.remove_handler(handler1, "resize")
    ee.submit({"event_type": "resize", "value": 2})
    ee.flush()
    assert values == [101, 201, 202]

    ee.remove_handler(handler2, "resize")
    ee.submit({"event_type": "resize", "value": 3})
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

    # handler3 goes first, the other two maintain order
    ee.add_handler(handler1, "resize")
    ee.add_handler(handler2, "resize")
    ee.add_handler(handler3, "resize", order=-1)

    ee.submit({"event_type": "resize", "value": 1})
    ee.flush()
    assert values == [301, 101, 201]


def test_events_duplicate_handler():

    ee = EventEmitter()

    values = []

    def handler(event):
        values.append(event["value"])

    # Registering for the same event_type twice just adds it once
    ee.add_handler(handler, "resize")
    ee.add_handler(handler, "resize")

    ee.submit({"event_type": "resize", "value": 1})
    ee.flush()
    assert values == [1]

    ee.remove_handler(handler, "resize")
    ee.submit({"event_type": "resize", "value": 2})
    ee.flush()
    assert values == [1]


def test_events_duplicate_handler_with_lambda():

    ee = EventEmitter()

    values = []

    def handler(event):
        values.append(event["value"])

    # Cannot discern now, these are two different handlers
    ee.add_handler(lambda e:handler(e), "resize")
    ee.add_handler(lambda e:handler(e), "resize")

    ee.submit({"event_type": "resize", "value": 1})
    ee.flush()
    assert values == [1, 1]

    ee.remove_handler(handler, "resize")
    ee.submit({"event_type": "resize", "value": 2})
    ee.flush()
    assert values == [1, 1, 2, 2]


def test_mini_benchmark():
    # Can be used to tweak internals of the EventEmitter and see the
    # effect on performance.

    ee = EventEmitter()

    def handler(event):
        pass

    t0 = time.perf_counter()
    for _ in range(1000):
        ee.add_handler(lambda e:handler(e), "resize", order=1)
        ee.add_handler(lambda e:handler(e), "resize", order=2)
    t1 = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(100):
        ee.submit({"event_type": "resize", "value": 2})
        ee.flush()
    t2 = time.perf_counter() - t0

    print(f"add_handler: {1000*t1:0.0f} ms, emit: {1000*t2:0.0f} ms")


if __name__ == "__main__":
    run_tests(globals())
