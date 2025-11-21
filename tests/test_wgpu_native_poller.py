import gc
import time
import queue

import wgpu
from wgpu.backends.wgpu_native._poller import PollThread, PollToken

from testutils import can_use_wgpu_lib, run_tests
from pytest import mark


def test_poll_thread():
    # A timeout to give polling thread time to progress. The GIL switches
    # threads about every 5ms, but in this cases likely faster, because it also switches
    # when it goes to sleep on a blocking call. So 50ms seems plenty.
    timeout = 0.05

    count = 0
    gpu_work_done_queue = queue.SimpleQueue()

    def reset():
        nonlocal count
        ref_count = count
        # Make sure the poller is not waiting in poll_func
        gpu_work_done_queue.put(None)
        gpu_work_done_queue.put(None)
        # Give it time
        time.sleep(timeout)
        # Check that it did not enter again, i.e. is waiting for tokens
        assert count == ref_count, "Looks like a token is still active"
        # Reset
        count = 0
        while True:
            try:
                gpu_work_done_queue.get(False)
            except queue.Empty:
                break

    def finish_tokens(*tokens):
        # This mimics the GPU finishing an async task, and invoking its
        # callback that sets the token to done.
        gpu_work_done_queue.put(None)
        for token in tokens:
            assert not token.is_done()
            token.set_done()

    def poll_func(block):
        # This mimics the wgpuDevicePoll.
        nonlocal count
        count += 1
        if block:
            gpu_work_done_queue.get()  # blocking
        else:
            try:
                gpu_work_done_queue.get(False)
            except queue.Empty:
                pass

    # Start the poller
    t = PollThread(poll_func)
    t.start()

    reset()

    # == Normal behavior

    token = t.get_token()
    assert isinstance(token, PollToken)
    time.sleep(timeout)
    assert count == 2

    finish_tokens(token)

    time.sleep(timeout)
    assert count == 2

    reset()

    # == Always at least one poll

    token = t.get_token()
    token.set_done()
    time.sleep(timeout)
    assert count in (1, 2)  # typically 1, but can sometimes be 2

    reset()

    # == Mark done through deletion

    token = t.get_token()
    time.sleep(timeout)
    assert count == 2

    finish_tokens()

    time.sleep(timeout)
    assert count == 3

    finish_tokens()

    time.sleep(timeout)
    assert count == 4

    del token
    gc.collect()
    gc.collect()

    finish_tokens()

    time.sleep(timeout)
    assert count == 4

    reset()

    # More tasks

    token1 = t.get_token()
    time.sleep(timeout)
    assert count == 2

    token2 = t.get_token()
    time.sleep(timeout)
    assert count == 2

    token3 = t.get_token()
    token4 = t.get_token()
    time.sleep(timeout)
    assert count == 2

    finish_tokens(token1)
    time.sleep(timeout)
    assert count == 3

    finish_tokens(token2, token3)
    time.sleep(timeout)
    assert count == 4

    finish_tokens()  # can actually bump more unrelated works
    finish_tokens()
    time.sleep(timeout)
    assert count == 6

    token5 = t.get_token()
    finish_tokens(token4)
    time.sleep(timeout)
    assert count == 7

    finish_tokens(token5)
    time.sleep(timeout)
    assert count == 8

    reset()

    # Shut it down

    t.stop()
    time.sleep(0.1)
    assert not t.is_alive()


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_poller_stops_when_device_gone():
    device = wgpu.gpu.request_adapter_sync().request_device_sync()

    t = device._poller
    assert t.is_alive()
    device.__del__()
    time.sleep(0.1)

    assert not t.is_alive()

    device = wgpu.gpu.request_adapter_sync().request_device_sync()

    t = device._poller
    assert t.is_alive()
    del device
    gc.collect()
    gc.collect()
    time.sleep(0.1)

    assert not t.is_alive()


if __name__ == "__main__":
    run_tests(globals())
