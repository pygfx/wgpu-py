import time
import asyncio


# todo: idea: a global loop proxy object that defers to any of the other loops
# would e.g. allow using glfw with qt together. Probably to weird a use-case for the added complexity.


class WgpuLoop:
    """Base class for different event-loop classes."""

    def call_soon(self, callback, *args):
        """Arrange for a callback to be called as soon as possible.

        Callbacks are called in the order in which they are registered.
        """
        self.call_later(0, callback, *args)

    def call_later(self, delay, callback, *args):
        """Arrange for a callback to be called after the given delay (in seconds)."""
        raise NotImplementedError()

    def poll(self):
        """Poll the underlying GUI toolkit for events.

        Some event loops (e.g. asyncio) are just that and dont have a GUI to update.
        """
        pass

    def run(self):
        """Enter the main loop."""
        raise NotImplementedError()

    def stop(self):
        """Stop the currently running event loop."""
        raise NotImplementedError()


class AnimationScheduler:
    """
    Some ideas:

    * canvas.events.connect("animate", callback)
    * canvas.animate.add_handler(1/30, callback)
    """

    def iter(self):
        # Something like this?
        for scheduler in all_schedulers:
            scheduler._event_emitter.submit_and_dispatch(event)


# todo: statistics on time spent doing what
