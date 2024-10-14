"""Implements an asyncio event loop."""

import asyncio

from .base import WgpuLoop


class AsyncioWgpuLoop(WgpuLoop):
    _the_loop = None

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
        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            pass
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

    def call_soon(self, callback, *args):
        self._loop.call_soon(callback, *args)

    def call_later(self, delay, callback, *args):
        self._loop.call_later(delay, callback, *args)

    def run(self):
        if self._loop.is_running():
            return  # Interactive mode!
        self._loop.run_forever()

    def stop(self):
        self._loop.stop()
