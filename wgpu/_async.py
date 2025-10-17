"""This module implements the core for the async support in wgpu."""

from __future__ import annotations

import sys
import time
import logging
import threading
from typing import Callable, Awaitable, Generator, Generic, TypeVar

import sniffio


logger = logging.getLogger("wgpu")


# The async_sleep and AsyncEvent are a copy of the implementation in rendercanvas.asyncs


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
    """A loop object must have (at least) this API.

    Rendercanvas loop objects do, asyncio.loop does too.
    """

    def call_soon(self, callback: Callable, *args: object):
        raise NotImplementedError()


def get_backoff_time_generator() -> Generator[float, None, None]:
    """Generates sleep-times, start at 0 then increasing to 100Hz and sticking there."""
    for _ in range(5):
        yield 0
    for i in range(1, 20):
        yield i / 2000.0  # ramp up from 0ms to 10ms
    while True:
        yield 0.01


class GPUPromise(Awaitable[AwaitedType], Generic[AwaitedType]):
    """A GPUPromise represents the eventual result of an asynchronous wgpu operation.

    A ``GPUPromise`` is a bit like an ``asyncio.Future``, but specific for wgpu, and with
    an API more similar to JavaScript's ``Promise``.

    Some methods of the wgpu API are asynchronous. They return a ``GPUPromise``,
    which provides a few different ways handle it:

    * It can be awaited using ``await promise``. This is the "cleanest" way, but
      can only be used from a co-routine (i.e. an async code path).
    * A callback can be registered using ``promise.then(callback)``, which will
      be called when the promise resolves. A new promise is returned.
    * You can sync-wait for it, using ``promise.sync_wait()``. This is simple, but
      makes code less portable (does not work on Pyodide backend) and potentially slower.

    A ``GPUPromise`` is in one of these states:

    * "pending": initial state, neither fulfilled nor rejected.
    * "pending-fulfilled": the input has been set, but the promise has not yet been resolved.
    * "pending-rejected": the error has been set, but the promise has not yet been resolved.
    * "fulfilled": meaning that the operation was completed successfully.
    * "rejected": meaning that the operation failed.
    """

    # We keep a set of unresolved promises, because whith using .then, noone else holds a ref to the promise
    _UNRESOLVED = set()

    def __init__(
        self,
        title: str,
        handler: Callable | None,
        *,
        loop: LoopInterface | None = None,
        poller: Callable | None = None,
        keepalive: object = None,
    ):
        """
        Arguments:
            title (str): The title of this promise, mostly for debugging purposes.
            handler (callable, optional): The function to turn promise input into the result. If None,
                the result will simply be the input.
            loop (LoopInterface, optional): A loop object that at least has a ``call_soon()`` method.
                If not given, this promise does not support .then() or pronise-chaining.
            poller (callable, optional): A function to call on a regular interval to poll internal systems
               (most likely the wgpu backend).
            keepalive (object, optional): Pass any data via this arg who's lifetime must be bound to the
                resolving of this prommise.

        """
        self._title = str(title)  # title for debugging
        self._handler = handler  # function to turn input into the result

        self._loop = loop  # Event loop instance, can be None
        self._poller = poller  # call to poll (process events)
        self._keepalive = keepalive  # just to keep something alive

        self._state = "pending"  # "pending", "pending-rejected", "pending-fulfilled", "rejected", "fulfilled"
        self._value = None  # The incoming value, final value, or error
        self._event = None  # AsyncEvent for __await__
        self._lock = threading.RLock()  # Allow threads to set the value
        self._done_callbacks = []
        self._error_callbacks = []
        self._UNRESOLVED.add(self)

    def __repr__(self):
        return f"<GPUPromise '{self._title}' {self._state} at {hex(id(self))}>"

    def __call__(self, callback):
        # Create new promise that invokes the callback
        self.then(callback)
        # Return original, because this is intended to be used as a decorator
        return callback

    def _wgpu_set_input(self, result):
        """Set the raw result, that will be passed through the handler to get the result.
        This method may be called from a different thread, or in another 'unexpected' moment. It does
        the minimal thing, and schedules a call to further process the result.
        """
        self._set_input(result, resolve_now=False)

    def _set_input(self, result: object, *, resolve_now=True) -> None:
        # Note that if resolve_now is True, it is assumed that this is the reference thread
        # and that this is a good time to handle the promise and invoke callbacks.

        # If the input is a promise, we need to wait for it, i.e. chain to self.
        if isinstance(result, GPUPromise):
            if self._loop is None:
                self._set_error("Cannot chain GPUPromise if the loop is not set.")
            else:
                result._chain(self)
            return

        with self._lock:
            if self._state != "pending":
                logger.warning(
                    f"Ignoring call to GPUPromise._set_input since promise state is {self._state!r}."
                )
                return
            self._state = "pending-fulfilled"
            self._value = result
            self._set_pending_resolved(resolve_now=resolve_now)

    def _wgpu_set_error(self, error: str | Exception) -> None:
        """Set the error, in case the promise could not be fulfilled.
        This method may be called from a different thread, or in another 'unexpected' moment. It does
        the minimal thing, and schedules a call to further process the result.
        """
        self._set_error(error, resolve_now=False)

    def _set_error(self, error: str | Exception, *, resolve_now=True) -> None:
        with self._lock:
            if self._state != "pending":
                logger.warning(
                    f"Ignoring call to GPUPromise._wgpu_set_error since promise state is {self._state!r}."
                )
                return
            if not isinstance(error, Exception):
                error = Exception(error)
            self._state = "pending-rejected"
            self._value = error
            self._set_pending_resolved(resolve_now=resolve_now)

    def _set_pending_resolved(self, *, resolve_now=False):
        """The promise received its input (or error), and now we need to handle it, then call callbacks etc."""
        # We can now drop the reference.
        self._UNRESOLVED.discard(self)
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
        """Finalize the promise, by calling the handler to get the result, and then invoking callbacks."""

        # We assume that this is only called from the appropriate thread,
        # and after the _wgpu_set_xxx is done, which is a reasonable assumption.

        # Finalize the value
        if self._state == "pending-fulfilled" and self._handler is not None:
            try:
                self._value = self._handler(self._value)
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
        self._handler = None
        self._poller = None
        self._keepalive = None
        # Resolve to the caller
        if self._state == "rejected":
            exception_in_promise = self._value
            raise exception_in_promise  # re-raising
        else:
            return self._value

    def sync_wait(self) -> AwaitedType:
        """Synchronously wait for the promise to resolve and return the result.

        Note that this method should be avoided in event callbacks, since it can
        make them slow.

        Note that this method may not be supported by all backends  (e.g. the
        upcoming JavaScript/Pyodide one), and using it will make your code less
        portable.
        """
        if self._state == "pending":
            if self._poller is None:
                raise RuntimeError(
                    "Cannot GPUPromise.sync_wait(), if the polling function is not set."
                )
            # Do small incremental sync naps. Other threads can run.
            # Note that time.sleep is accurate (does not suffer from the inaccuracy issue on Windows).
            sleep_gen = get_backoff_time_generator()
            self._poller()
            while self._state == "pending":
                time.sleep(next(sleep_gen))
                self._poller()

        return self._resolve()  # returns result if fulfilled or raise error if rejected

    def _chain(self, to_promise: GPUPromise):
        with self._lock:
            self._done_callbacks.append(to_promise._set_input)
            self._error_callbacks.append(to_promise._set_error)
            if not self._state.startswith("pending"):
                self._resolve()

    def then(
        self,
        callback: Callable[[AwaitedType], None],
        error_callback: Callable[[Exception], None] | None = None,
        title: str | None = None,
    ):
        """Set a callback that will be called when the promise is fulfilled.

        The callback will receive one argument: the result of the promise.
        """
        if self._loop is None:
            raise RuntimeError("Cannot use GPUPromise.then() if the loop is not set.")
        if not callable(callback):
            raise TypeError(
                f"GPUPromise.then() got a callback that is not callable: {callback!r}"
            )

        # Get title for the new promise
        if title is not None:
            title = str(title)
        else:
            try:
                callback_name = callback.__name__
            except Exception:
                callback_name = str(callback)
            title = self._title + " -> " + callback_name

        # Create new promise
        new_promise = self.__class__(
            title, callback, loop=self._loop, poller=self._poller
        )
        self._chain(new_promise)

        if error_callback is not None:
            self.catch(error_callback)

        return new_promise

    def catch(self, callback: Callable[[Exception], None] | None):
        """Set a callback that will be called when the promise is rejected.

        The callback will receive one argument: the error object.
        """
        if self._loop is None:
            raise RuntimeError("Cannot use GPUPromise.catch() if the loop is not set.")
        if not callable(callback):
            raise TypeError(
                f"GPUPromise.catch() got a callback that is not callable: {callback!r}"
            )

        # Get title for the new promise
        title = "Catcher for " + self._title

        # Create new promise
        new_promise = self.__class__(
            title, callback, loop=self._loop, poller=self._poller
        )

        # Custom chain
        with self._lock:
            self._error_callbacks.append(new_promise._set_input)
            if not self._state.startswith("pending"):
                self._resolve()

        return new_promise

    def __await__(self):
        if self._loop is None:
            # An async busy loop
            async def awaiter():
                if self._state == "pending":
                    # backoff_time_generator = self._get_backoff_time_generator()
                    if self._poller is None:
                        raise RuntimeError(
                            "Cannot await a GPUPromise if neither the loop nor the poller are set."
                        )
                    # Do small incremental async naps. Other tasks and threads can run.
                    # Note that async sleep, with sleep_time > 0, is inaccurate on Windows.
                    sleep_gen = get_backoff_time_generator()
                    self._poller()
                    while self._state == "pending":
                        await async_sleep(next(sleep_gen))
                        self._poller()
                return self._resolve()

        else:
            # Using a signal
            with self._lock:
                if self._event is None:
                    self._event = AsyncEvent()
                    if self._state != "pending":
                        self._event.set()

            async def awaiter():
                await self._event.wait()
                return self._resolve()

        return (yield from awaiter().__await__())
