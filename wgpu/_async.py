import sys
import time
import logging
import threading
from typing import Callable, Awaitable, Generic, TypeVar

import sniffio


logger = logging.getLogger("wgpu")


# TODO: make a public module? wgpu.async

# TODO: this module looks a lot like rendercanvas.asyncs. Say/do something about that.

# TODO: GPUFuture or GPUPromise; Python API or JS?
# Leaning towards the JS
#
# JS:
# promise.then(lambda result: ...)
# promise.then(handle_result, handle_exception)
# promise.catch(handle_exception)
# primise.finally()
#
# Python:
# future.result()
# future.set_result()
# future.set_exception()
# future.done()
# future.cancelled()
# future.add_done_callback(lambda future: ...)
# future.remove_done_callback()
# future.cancel()
# future.exception()
# future.get_loop()


async def async_sleep(delay):
    """Async sleep that uses sniffio to be compatible with asyncio, trio, rendercanvas.utils.asyncadapter, and possibly more."""
    libname = sniffio.current_async_library()
    sleep = sys.modules[libname].sleep
    await sleep(delay)


class AsyncEvent:
    """Generic async event object using sniffio. Works with trio, asyncio and rendercanvas-native."""

    def __new__(cls):
        libname = sniffio.current_async_library()
        Event = sys.modules[libname].Event  # noqa
        return Event()


AwaitedType = TypeVar("AwaitedType")


class LoopInterface:
    def call_soon(self, callbacl: Callable, *args: object):
        raise NotImplementedError()


class GPUPromise(Awaitable[AwaitedType], Generic[AwaitedType]):
    """A GPUPromise represents the eventual result of an asynchronous wgpu operation.

    A ``GPUPromise`` is a bit like an ``asyncio.Future``, but specific for wgpu, and with
    an API more similar to JavaScript's ``Promise``.

    Some methods of the wgpu API are asynchronous. They return a ``GPUPromise``,
    which provides a few different ways handle it:

    * It can be awaited using ``await future``. This is the "cleanest" way, but
      can only be used from a co-routine (i.e. an async code path).
    * A callback can be registered using ``future.then(callback)``, which will
      be called when the future resolves.
    * You can sync-wait for it, using ``future.wait()``. This is simple, but
      makes code less portable and potentially slower.

    A ``GPUPromise`` is in one of these states:

    * pending: initial state, neither fulfilled nor rejected.
    * fulfilled: meaning that the operation was completed successfully.
    * rejected: meaning that the operation failed.
    """

    # We keep a set of unresolved promises, because whith using .then, noone else holds a ref to the promise
    _UNRESOLVED = set()

    def __init__(
        self,
        title: str,
        loop: LoopInterface | None,
        finalizer: Callable | None,
        *,
        poller: Callable | None = None,
        keepalive: object = None,
    ):
        self._title = str(title)  # title for debugging
        self._loop = loop  # Event loop instance, can be None
        self._finalizer = finalizer  # function to finish the result
        self._poller = poller  # call to poll (process events)
        self._keepalive = keepalive  # just to keep something alive

        self._state = "pending"  # "pending", "pending-rejected", "pending-fulfilled", "rejected", "fulfilled"
        self._value = None  # The incoming value, final value, or error
        self._event = None  # AsyncEvent for __await__
        self._lock = threading.RLock()  # Allow threads to set the value
        self._done_callbacks = []
        self._error_callbacks = []
        GPUPromise._UNRESOLVED.add(self)

    def __repr__(self):
        value_repr = ""
        if self._state == "fulfilled":
            value_repr = repr(self._value).split("\n", 1)[0]
            if len(value_repr) > 30:
                value_repr = value_repr[:29] + "…"
            value_repr = f"'{value_repr}'"
        return (
            f"<GPUPromise {self._title} {self._state} {value_repr} at {hex(id(self))}>"
        )

    def __call__(self, callback):
        return self.then(callback)

    def _wgpu_set_raw_result(self, result):
        """Set the raw result, that will be passed through the finalizer to get the actual result.
        This method may be called from a different thread, or in another 'unexpected' moment. It does
        the minimal thing, and schedules a call to further process the result.
        """
        self._set_raw_result(result, resolve_now=False)

    def _set_raw_result(self, result: object, *, resolve_now=True) -> None:
        with self._lock:
            if self._state != "pending":
                logger.warning(
                    "Ignoring call to GPUPromise._set_raw_result since promise state is {self._state!r}."
                )
                return
            self._state = "pending-fulfilled"
            self._value = result
            self._set_raw_resolved(resolve_now=resolve_now)

    def _wgpu_set_error(self, error: Exception) -> None:
        """Set the error, in case the promise could not be fulfilled.
        This method may be called from a different thread, or in another 'unexpected' moment. It does
        the minimal thing, and schedules a call to further process the result.
        """
        self._set_error(error, resolve_now=False)

    def _set_error(self, error: Exception, *, resolve_now=True) -> None:
        with self._lock:
            if self._state != "pending":
                logger.warning(
                    "Ignoring call to GPUPromise._wgpu_set_error since promise state is {self._state!r}."
                )
                return
            if not isinstance(error, Exception):
                error = Exception(error)
            self._state = "pending-rejected"
            self._value = error
            self._set_raw_resolved(resolve_now=resolve_now)

    def _set_raw_resolved(self, *, resolve_now=False):
        """The promise is fulfilled, but now we need to handle it, like call callbacks etc."""
        # We can now drop the reference.
        GPUPromise._UNRESOLVED.discard(self)
        # Do or schedule a call to resolve.
        if resolve_now:
            self._resolve_callback()
        elif self._loop is not None:
            self._loop.call_soon(self._resolve_callback)
        # Allow tasks that await this promise to continue. Do this last, since
        # it allows any waiting tasks to continue. These taks are assumed to be
        # on the 'reference' thread, but *this* may be a different thread.
        if self._event is not None:
            self._event.set()

    def _resolve_callback(self):
        # The callback may already be resolved
        if self._state.startswith("pending-"):
            self._resolve()

    def _resolve(self):
        """Finalize the promise."""

        # We assume that this is only called from the appropriate thread,
        # and after the _wgpu_set_xxx is done, which is a reasonable assumption.

        # Finalize the value
        if self._state == "pending-fulfilled" and self._finalizer is not None:
            try:
                self._value = self._finalizer(self._value)
            except Exception as err:
                self._state = "rejected"
                self._value = err
        # Schedule the callbacks
        if self._state.endswith("rejected"):
            error = self._value
            for cb in self._error_callbacks:
                self._loop.call_soon(cb, error)
        elif self._state.endswith("fulfilled"):
            result = self._value
            for cb in self._done_callbacks:
                self._loop.call_soon(cb, result)
        # New state
        self._state = self._state.replace("pending-", "")
        # Clean up
        self._error_callbacks = []
        self._done_callbacks = []
        self._finalizer = None
        self._poller = None
        self._keepalive = None
        # Resolve to the caller
        if self._state == "rejected":
            raise self._value  # type:ignore
        else:
            return self._value

    def sync_wait(self) -> AwaitedType:
        """Synchronously wait for the future to resolve and return the result.

        Note that this method should be avoided in event callbacks, since it can
        make them slow.

        Note that this method may not be supported by all backends  (e.g. the
        upcoming JavaScript/Pyodide one), and using it will make your code less
        portable.
        """
        if self._state == "pending":
            if self._poller is None:
                raise RuntimeError("Expected callback to have already happened")
            self._poller()
            while self._state == "pending":
                time.sleep(0)
                self._poller()

        return self._resolve()  # returns result if fulfilled or raise error if rejected

    def then(self, callback: Callable[[AwaitedType], None]):
        """Set a callback that will be called when the future resolves.

        The callback will receive one argument: the result of the future.
        """
        if self._loop is None:
            raise RuntimeError("Cannot use GPUPromise.then() if loop is not set.")
        if callable(callback):
            self._callback = callback
        else:
            raise TypeError(
                f"GPUPromise.then() got a callback that is not callable: {callback!r}"
            )

        # Create proxy promise
        title = self._title + " -> " + str(callback)
        new_promise = GPUPromise(title, self._loop, callback, poller=self._poller)

        with self._lock:
            self._done_callbacks.append(new_promise._set_raw_result)
            self._error_callbacks.append(new_promise._set_error)
            if not self._state.startswith("pending"):
                self._resolve()

        # TODO: allow calling multiple times
        # TODO: allow calling after being resolved -> tests!
        # TODO: return another promise, so we can do chaining? Or maybe not interesting for this use-case...

        return new_promise

    def __await__(self):
        if self._loop is None:
            raise RuntimeError("Cannot await a GPUPromise if loop is not set.")
        with self._lock:
            if self._event is None:
                self._event = AsyncEvent()
                if self._state != "pending":
                    self._event.set()

        async def wrapper():
            await self._event.wait()
            return self._resolve()

        return (yield from wrapper().__await__())
