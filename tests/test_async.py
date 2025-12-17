import sys
import time
import types
import asyncio
import threading

import trio
import anyio
from pytest import mark, raises

from rendercanvas.raw import RawLoop
import wgpu.utils
from testutils import can_use_wgpu_lib, run_tests
from wgpu import GPUDevice, MapMode, TextureFormat
from wgpu._async import (
    GPUPromise as BaseGPUPromise,
    detect_current_call_soon_threadsafe,
    detect_current_async_lib,
)


class GPUPromise(BaseGPUPromise):
    # Subclass with each own set of unresolved promise instances
    _UNRESOLVED = set()

    def _sync_wait(self):
        # Same implementation as the wgpu_native backend.
        # If we have a test that has not polling thread, and sync_wait() is called
        # when the promise is still pending, this will hang.
        self._thread_event.wait()


class SillyLoop:
    def __init__(self):
        self._pending_calls = []
        self.errors = []

    def call_soon(self, f, *args):  # and its threadsafe
        self._pending_calls.append((f, args))

    def process_events(self):
        for p in list(GPUPromise._UNRESOLVED):
            if p._title == "test" and p._state == "pending":
                p._wgpu_set_input(7)
        while self._pending_calls:
            f, args = self._pending_calls.pop(0)
            try:
                f(*args)
            except Exception as err:
                self.errors.append(err)


def test_promise_basics():
    promise = GPUPromise("foobar", None)

    assert "foobar" in repr(promise)

    assert "pending" in repr(promise)
    assert "fulfilled" not in repr(promise)
    assert "rejected" not in repr(promise)

    promise._wgpu_set_input(42)

    assert "pending-fulfilled" in repr(promise)
    assert "rejected" not in repr(promise)

    result = promise.sync_wait()

    assert result == 42
    assert "pending" not in repr(promise)
    assert "fulfilled" in repr(promise)
    assert "rejected" not in repr(promise)

    # err

    promise = GPUPromise("foobar", None)
    promise._wgpu_set_error("woops")
    assert "pending-rejected" in repr(promise)
    assert "fulfilled" not in repr(promise)

    with raises(Exception) as err:
        result = promise.sync_wait()

    assert err.match("woops")
    assert "pending" not in repr(promise)
    assert "fulfilled" not in repr(promise)
    assert "rejected" in repr(promise)


# %%%%% Low level


def test_async_low_level_none():
    flag = []

    flag.append(detect_current_async_lib())
    flag.append(detect_current_call_soon_threadsafe())

    assert flag[0] is None
    assert flag[1] is None


def test_async_low_level_rendercanvas_asyncadapter():
    loop = RawLoop()

    flag = []

    async def task():
        # Our methods
        flag.append(detect_current_async_lib())
        flag.append(detect_current_call_soon_threadsafe())
        # Test that the fast-path works
        flag.append(sys.get_asyncgen_hooks()[0].__self__.call_soon_threadsafe)
        loop.stop()

    loop.add_task(task)
    loop.run()

    assert flag[0] == "rendercanvas.utils.asyncadapter"
    assert callable(flag[1])
    assert flag[1].__name__ == "call_soon_threadsafe"
    assert flag[1].__func__ is flag[2].__func__


def test_async_low_level_asyncio():
    flag = []

    async def task():
        # Our methods
        flag.append(detect_current_async_lib())
        flag.append(detect_current_call_soon_threadsafe())
        # Test that the fast-path works
        flag.append(sys.get_asyncgen_hooks()[0].__self__.call_soon_threadsafe)

    asyncio.run(task())

    assert flag[0] == "asyncio"
    assert callable(flag[1])
    assert flag[1].__name__ == "call_soon_threadsafe"
    assert flag[1].__func__ is flag[2].__func__


def test_async_low_level_trio():
    flag = []

    async def task():
        flag.append(detect_current_async_lib())
        flag.append(detect_current_call_soon_threadsafe())

    trio.run(task)

    assert flag[0] == "trio"
    assert callable(flag[1])
    assert flag[1].__name__ == "run_sync_soon"


def test_async_low_level_custom1():
    # Simplest custom approach. Detection at module level.

    mod = types.ModuleType("wgpu_async_test_module")
    sys.modules[mod.__name__] = mod
    code = """if True:

    def call_soon_threadsafe(callbacl):
        pass

    def fake_asyncgen_hook(agen):
        pass
    """
    exec(code, mod.__dict__)

    flag = []

    old_hooks = sys.get_asyncgen_hooks()
    sys.set_asyncgen_hooks(mod.fake_asyncgen_hook)

    try:
        flag.append(detect_current_async_lib())
        flag.append(detect_current_call_soon_threadsafe())
    finally:
        sys.set_asyncgen_hooks(*old_hooks)

    assert flag[0] == mod.__name__
    assert flag[1] is mod.call_soon_threadsafe


def test_async_low_level_custom2():
    # Even better, call_soon_threadsafe is attr of the same object that asyncgen hook is a method of.
    # This takes the fast path!

    mod = types.ModuleType("wgpu_async_test_module")
    sys.modules[mod.__name__] = mod
    code = """if True:

    class Loop:
        def call_soon_threadsafe(callbacl):
            pass

        def fake_asyncgen_hook(agen):
            pass
    loop = Loop()
    """
    exec(code, mod.__dict__)

    flag = []

    old_hooks = sys.get_asyncgen_hooks()
    sys.set_asyncgen_hooks(mod.loop.fake_asyncgen_hook)

    try:
        flag.append(detect_current_async_lib())
        flag.append(detect_current_call_soon_threadsafe())
    finally:
        sys.set_asyncgen_hooks(*old_hooks)

    assert flag[0] == mod.__name__
    assert flag[1].__func__ is mod.loop.call_soon_threadsafe.__func__


def test_async_low_level_custom3():
    # The somewhat longer route. This is also the fallback for asyncio,
    # in case they change something that kills the fast-path for asyncio.
    # (the fast path being sys.get_asyncgen_hooks()[0].__self__.call_soon_threadsafe)

    mod = types.ModuleType("wgpu_async_test_module")
    sys.modules[mod.__name__] = mod
    code = """if True:

    def fake_asyncgen_hook(agen):
        pass
    def get_running_loop():
        return loop
    class Loop:
        def call_soon_threadsafe(callbacl):
            pass
    loop = Loop()
    """
    exec(code, mod.__dict__)

    flag = []

    old_hooks = sys.get_asyncgen_hooks()
    sys.set_asyncgen_hooks(mod.fake_asyncgen_hook)

    try:
        flag.append(detect_current_async_lib())
        flag.append(detect_current_call_soon_threadsafe())
    finally:
        sys.set_asyncgen_hooks(*old_hooks)

    assert flag[0] == mod.__name__
    assert flag[1].__func__ is mod.loop.call_soon_threadsafe.__func__


# %%%%% Promise using sync_wait


def run_in_thread(callable):
    t = threading.Thread(target=callable)
    t.start()


def test_promise_sync_simple():
    @run_in_thread
    def poller():
        time.sleep(0.1)
        promise._wgpu_set_input(42)

    promise = GPUPromise("test", None)

    result = promise.sync_wait()
    assert result == 42


def test_promise_sync_normal():
    def handler(input):
        return input * 2

    @run_in_thread
    def poller():
        time.sleep(0.1)
        promise._wgpu_set_input(42)

    promise = GPUPromise("test", handler)

    result = promise.sync_wait()
    assert result == 84


def test_promise_sync_fail1():
    def handler(input):
        return input * 2

    @run_in_thread
    def poller():
        time.sleep(0.1)
        promise._wgpu_set_error(ZeroDivisionError())

    promise = GPUPromise("test", handler)

    with raises(ZeroDivisionError):
        promise.sync_wait()


def test_promise_sync_fail2():
    def handler(input):
        return input / 0

    @run_in_thread
    def poller():
        time.sleep(0.1)
        promise._wgpu_set_input(42)

    promise = GPUPromise("test", handler)

    with raises(ZeroDivisionError):
        promise.sync_wait()


# %% Promise using await with poll and loop


@mark.anyio
async def test_promise_async_poll_simple():
    @run_in_thread
    def poller():
        time.sleep(0.1)
        promise._wgpu_set_input(42)

    promise = GPUPromise("test", None)

    result = await promise
    assert result == 42


@mark.anyio
async def test_promise_async_poll_normal():
    def handler(input):
        return input * 2

    @run_in_thread
    def poller():
        time.sleep(0.1)
        promise._wgpu_set_input(42)

    promise = GPUPromise("test", handler)

    result = await promise
    assert result == 84


@mark.anyio
async def test_promise_async_poll_fail1():
    def handler(input):
        return input * 2

    @run_in_thread
    def poller():
        time.sleep(0.1)
        promise._wgpu_set_error(ZeroDivisionError())

    promise = GPUPromise("test", handler)

    with raises(ZeroDivisionError):
        await promise


@mark.anyio
async def test_promise_async_poll_fail2():
    def handler(input):
        return input / 0

    @run_in_thread
    def poller():
        time.sleep(0.1)
        promise._wgpu_set_input(42)

    promise = GPUPromise("test", handler)

    with raises(ZeroDivisionError):
        await promise


@mark.anyio
async def test_promise_async_loop_simple():
    loop = SillyLoop()

    promise = GPUPromise("test", None, _call_soon_threadsafe=loop.call_soon)

    loop.process_events()
    result = await promise
    assert result == 7


@mark.anyio
async def test_promise_async_loop_normal():
    loop = SillyLoop()

    def handler(input):
        return input * 2

    promise = GPUPromise("test", handler, _call_soon_threadsafe=loop.call_soon)

    loop.process_events()
    result = await promise
    assert result == 14


@mark.anyio
async def test_promise_async_loop_fail2():
    loop = SillyLoop()

    def handler(input):
        return input / 0

    promise = GPUPromise("test", handler, _call_soon_threadsafe=loop.call_soon)

    loop.process_events()
    with raises(ZeroDivisionError):
        await promise


# %%%%% Promise using callbacks


def test_promise_then_need_loop():
    result = None

    def callback(r):
        nonlocal result
        result = r

    promise = GPUPromise("test", None)

    with raises(RuntimeError):  # cannot poll without a loop
        promise.then(callback)


def test_promise_then_simple():
    loop = SillyLoop()

    result = None

    def callback(r):
        nonlocal result
        result = r

    promise = GPUPromise("test", None, _call_soon_threadsafe=loop.call_soon)

    promise.then(callback)
    loop.process_events()
    assert result == 7


def test_promise_then_normal():
    loop = SillyLoop()

    result = None

    def callback(r):
        nonlocal result
        result = r

    def handler(input):
        return input * 2

    promise = GPUPromise("test", handler, _call_soon_threadsafe=loop.call_soon)

    promise.then(callback)
    loop.process_events()
    assert result == 14


def test_promise_then_fail2():
    loop = SillyLoop()

    result = None
    error = None

    def callback(r):
        nonlocal result
        result = r

    def err_callback(e):
        nonlocal error
        error = e

    def handler(input):
        return input / 0

    promise = GPUPromise("test", handler, _call_soon_threadsafe=loop.call_soon)

    promise.then(callback, err_callback)
    loop.process_events()
    assert result is None
    assert isinstance(error, ZeroDivisionError)


# %%%%% Chaining


def test_promise_chaining_basic():
    loop = SillyLoop()

    class MyPromise(GPUPromise):
        pass

    result = None

    def callback1(r):
        nonlocal result
        result = r

    promise = MyPromise("test", None, _call_soon_threadsafe=loop.call_soon)

    p = promise.then(callback1)
    loop.process_events()
    assert result == 7

    # New prommise is of same class
    assert isinstance(p, MyPromise)

    # Repr is a path
    assert "test -> callback1" in repr(p)

    # Unless overriden
    p2 = promise.then(callback1, title="foobar")
    assert "test -> callback1" not in repr(p2)
    assert "foobar" in repr(p2)


def test_promise_chaining_simple():
    loop = SillyLoop()

    result = None

    def callback1(r):
        return r * 3

    def callback2(r):
        return r + 2

    def callback3(r):
        nonlocal result
        result = r

    promise = GPUPromise("test", None, _call_soon_threadsafe=loop.call_soon)

    p = promise.then(callback1).then(callback2).then(callback3)
    assert isinstance(p, GPUPromise)

    loop.process_events()
    assert result == 7 * 3 + 2


def test_promise_chaining_fail1():
    loop = SillyLoop()

    result = None
    error = None

    def callback1(r):
        return r * 3 / 0

    def callback2(r):
        return r + 2

    def callback3(r):
        nonlocal result
        result = r

    def err_callback(e):
        nonlocal error
        error = e

    promise = GPUPromise("test", None, _call_soon_threadsafe=loop.call_soon)

    p = promise.then(callback1).then(callback2).then(callback3, err_callback)
    assert isinstance(p, GPUPromise)

    loop.process_events()
    assert result is None
    assert isinstance(error, ZeroDivisionError)


def test_promise_chaining_fail2():
    loop = SillyLoop()

    result = None
    error = None

    def callback1(r):
        return r * 3 / 0

    def callback2(r):
        return r + 2

    def callback3(r):
        nonlocal result
        result = r / 0

    def err_callback(e):
        nonlocal error
        error = e

    promise = GPUPromise("test", None, _call_soon_threadsafe=loop.call_soon)

    p = promise.then(callback1).then(callback2).then(callback3, err_callback)
    assert isinstance(p, GPUPromise)

    loop.process_events()
    assert result is None
    assert isinstance(error, ZeroDivisionError)


def test_promise_chaining_multi():
    loop = SillyLoop()

    results = []

    def callback1(r):
        results.append(r)

    def callback2(r):
        results.append(r * 2)

    def callback3(r):
        results.append(r * 3)

    promise = GPUPromise("test", None, _call_soon_threadsafe=loop.call_soon)

    promise.then(callback1)
    promise.then(callback2)
    promise.then(callback3)
    promise.then(callback1)

    loop.process_events()
    assert results == [7, 14, 21, 7]


def test_promise_chaining_after_resolve():
    loop = SillyLoop()

    results = []

    def callback1(r):
        results.append(r)

    promise = GPUPromise("test", None, _call_soon_threadsafe=loop.call_soon)

    # Adding handler has no result, because promise is not yet resolved.
    promise.then(callback1)
    assert results == []

    # Resolving adds the result
    loop.process_events()
    assert results == [7]

    # Resolves once :)
    loop.process_events()
    loop.process_events()
    assert results == [7]

    # But we can add a handler after it has resolved
    promise.then(callback1)
    assert results == [7]
    loop.process_events()
    assert results == [7, 7]


def test_promise_chaining_with_promises():
    # If the result if a promise is another promise, then the chained
    # (wrapper) promise will wait for that promise to get the result.
    loop = SillyLoop()

    result = None

    def callback1(r):
        return GPUPromise("test", lambda _: r * 3, _call_soon_threadsafe=loop.call_soon)

    def callback2(r):
        return GPUPromise("test", lambda _: r + 2, _call_soon_threadsafe=loop.call_soon)

    def callback3(r):
        nonlocal result
        result = r

    promise = GPUPromise("test", None, _call_soon_threadsafe=loop.call_soon)

    p = promise.then(callback1).then(callback2).then(callback3)
    assert isinstance(p, GPUPromise)

    # Need more processing events to set the input of the new promises
    loop.process_events()
    loop.process_events()
    loop.process_events()
    assert result == 7 * 3 + 2


# %%%%% Other


def test_promise_decorator():
    loop = SillyLoop()

    result = None

    def handler(input):
        return input * 2

    promise = GPUPromise("test", handler, _call_soon_threadsafe=loop.call_soon)

    @promise
    def decorated(r):
        nonlocal result
        result = r

    loop.process_events()
    assert result == 14

    # Decorating returns original, not a new promise like then()
    assert not isinstance(decorated, GPUPromise)
    assert promise(decorated) is decorated


# %%%%% Test the async methods


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
@mark.anyio
async def test_enumerate_adapters_async():
    adapters = await wgpu.gpu.enumerate_adapters_async()
    assert len(adapters) > 0
    for adapter in adapters:
        device = await adapter.request_device_async()
        assert isinstance(device, GPUDevice)


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
@mark.anyio
async def test_request_device_async():
    adapter = await wgpu.gpu.request_adapter_async(power_preference="high-performance")
    device = await adapter.request_device_async()
    assert device is not None


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
@mark.anyio
async def test_buffer_map_async():
    device = wgpu.utils.get_default_device()

    data = b"1" * 10000
    buffer1 = device.create_buffer(size=len(data), usage="MAP_WRITE|COPY_SRC")
    buffer2 = device.create_buffer(size=len(data), usage="MAP_READ|COPY_DST")
    await buffer1.map_async(MapMode.WRITE)
    buffer1.write_mapped(data)
    buffer1.unmap()

    command_encoder = device.create_command_encoder()
    command_encoder.copy_buffer_to_buffer(buffer1, 0, buffer2, 0, len(data))
    device.queue.submit([command_encoder.finish()])

    await buffer2.map_async(MapMode.READ)
    data2 = buffer2.read_mapped()
    buffer2.unmap()

    assert bytes(data2) == data


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
@mark.anyio
async def make_pipeline_async():
    device = wgpu.utils.get_default_device()

    shader_source = """
        @vertex
        fn vertex_main() -> @builtin(position) vec4f {
            return vec4f(0, 0, 0, 1.);
        }

        @compute @workgroup_size(1)
        fn compute_main() { }
    """

    shader = device.create_shader_module(code=shader_source)

    results = [None, None]
    async with anyio.create_task_group() as tg:
        # It's unfortunate anyio doesn't have async.gather. This code would just be
        # compute_pipeline, render_pipeline = asyncio.gather(.....)
        async def create_compute_pipeline():
            results[0] = await device.create_compute_pipeline_async(
                layout="auto", compute={"module": shader}
            )

        async def create_render_pipeline():
            results[1] = await device.create_render_pipeline_async(
                layout="auto",
                vertex={
                    "module": shader,
                },
                depth_stencil={"format": TextureFormat.rgba8unorm},
            )

        tg.start_soon(create_compute_pipeline)
        tg.start_soon(create_render_pipeline)

    compute_pipeline, render_pipeline = results
    assert compute_pipeline is not None
    assert render_pipeline is not None

    command_encoder = device.create_command_encoder()
    compute_pass = command_encoder.begin_compute_pass()
    compute_pass.set_pipeline(compute_pipeline)
    compute_pass.dispatch_workgroups(10, 10)
    compute_pass.end()
    device.queue.submit([command_encoder.finish()])
    await device.queue.on_submitted_work_done_async()


if __name__ == "__main__":
    run_tests(globals())
