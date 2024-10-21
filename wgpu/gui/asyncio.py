"""Implements an asyncio event loop."""

# This is used for GUI backends that don't have an event loop by themselves, like glfw.
# Would be nice to also allow a loop based on e.g. Trio. But we can likely fit that in
# when the time comes.

import asyncio

from .base import WgpuLoop, WgpuTimer


class AsyncioWgpuTimer(WgpuTimer):
    """Wgpu timer based on asyncio."""

    _handle = None

    def _start(self):
        def tick():
            self._handle = None
            self._tick()

        if self._handle is not None:
            self._handle.cancel()
        asyncio_loop = self._loop._loop
        self._handle = asyncio_loop.call_later(self._interval, tick)

    def _stop(self):
        if self._handle:
            self._handle.cancel()
            self._handle = None


class AsyncioWgpuLoop(WgpuLoop):
    _TimerClass = AsyncioWgpuTimer
    _the_loop = None
    _is_interactive = False

    @property
    def _loop(self):
        if self._the_loop is None:
            self._the_loop = self._get_loop()
        return self._the_loop

    def _get_loop(self):
        try:
            return asyncio.get_running_loop()
        except Exception:
            pass
        # todo: get_event_loop is on a deprecation path.
        # but there still is `set_event_loop()`  so I'm a bit confused
        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            pass
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

    def _run(self):
        if self._loop.is_running():
            self._is_interactive = True
        else:
            self._is_interactive = False
            self._loop.run_forever()

    def _stop(self):
        if not self._is_interactive:
            self._loop.stop()

    def _call_soon(self, callback, *args):
        self._loop.call_soon(callback, *args)
