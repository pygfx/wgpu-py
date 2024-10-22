"""
Implemens loop mechanics: The base timer, base loop, and scheduler.
"""

import time
import weakref

from ._gui_utils import log_exception
from ..enums import Enum

# Note: technically, we could have a global loop proxy object that defers to any of the other loops.
# That would e.g. allow using glfw with qt together. Probably to too weird use-case for the added complexity.


class WgpuTimer:
    """Base class for a timer objects."""

    _running_timers = set()

    def __init__(self, loop, callback, *args, one_shot=False):
        # The loop arg is passed as an argument, so that the Loop object itself can create a timer.
        self._loop = loop
        # Check callable
        if not callable(callback):
            raise TypeError("Given timer callback is not a callable.")
        self._callback = callback
        self._args = args
        # Internal variables
        self._one_shot = bool(one_shot)
        self._interval = 0.0
        self._expect_tick_at = None

    def start(self, interval):
        """Start the timer with the given interval.

        When the interval has passed, the callback function will be called,
        unless the timer is stopped earlier.

        When the timer is currently running, it is first stopped and then
        restarted.
        """
        if self.is_running:
            self._stop()
        WgpuTimer._running_timers.add(self)
        self._interval = float(interval)
        self._expect_tick_at = time.perf_counter() + self._interval
        self._start()

    def stop(self):
        """Stop the timer.

        If the timer is currently running, it is stopped, and the
        callback is *not* called. If the timer is currently not running,
        this method does nothing.
        """
        WgpuTimer._running_timers.discard(self)
        self._expect_tick_at = None
        self._stop()

    def _tick(self):
        """The implementations must call this method."""
        # Stop or restart
        if self._one_shot:
            WgpuTimer._running_timers.discard(self)
            self._expect_tick_at = None
        else:
            self._expect_tick_at = time.perf_counter() + self._interval
            self._start()
        # Callback
        with log_exception("Timer callback error"):
            self._callback(*self._args)

    @property
    def time_left(self):
        """The expected time left before the callback is called.

        None means that the timer is not running. The value can be negative
        (which means that the timer is late).
        """
        if self._expect_tick_at is None:
            return None
        else:
            return self._expect_tick_at - time.perf_counter()

    @property
    def is_running(self):
        """Whether the timer is running."""
        return self._expect_tick_at is not None

    @property
    def is_one_shot(self):
        """Whether the timer is one-shot or continuous."""
        return self._one_shot

    def _start(self):
        """For the subclass to implement:

        * Must schedule for ``self._tick`` to be called in ``self._interval`` seconds.
        * Must call it exactly once (the base class takes care of repeating the timer).
        * When ``self._stop()`` is called before the timer finished, the call to ``self._tick()`` must be cancelled.
        """
        raise NotImplementedError()

    def _stop(self):
        """For the subclass to implement:

        * If the timer is running, cancel the pending call to ``self._tick()``.
        * Otherwise, this should do nothing.
        """
        raise NotImplementedError()


class WgpuLoop:
    """Base class for event-loop objects."""

    _TimerClass = None  # subclases must set this

    def __init__(self):
        self._schedulers = set()
        self._stop_when_no_canvases = False
        self._gui_timer = self._TimerClass(self, self._tick, one_shot=False)

    def _register_scheduler(self, scheduler):
        # Gets called whenever a canvas in instantiated
        self._schedulers.add(scheduler)
        self._gui_timer.start(0.1)  # (re)start our internal timer

    def _tick(self):
        # Keep the GUI alive on every tick
        self._wgpu_gui_poll()

        # Check all schedulers
        schedulers_to_close = []
        for scheduler in self._schedulers:
            if scheduler._get_canvas() is None:
                schedulers_to_close.append(scheduler)

        # Forget schedulers that no longer have an live canvas
        for scheduler in schedulers_to_close:
            self._schedulers.discard(scheduler)

        # Check whether we must stop the loop
        if self._stop_when_no_canvases and not self._schedulers:
            self.stop()

    def call_soon(self, callback, *args):
        """Arrange for a callback to be called as soon as possible.

        The callback will be called in the next iteration of the event-loop,
        but other pending events/callbacks may be handled first. Returns None.
        """
        self._call_soon(callback, *args)

    def call_later(self, delay, callback, *args):
        """Arrange for a callback to be called after the given delay (in seconds).

        Returns a timer object (in one-shot mode) that can be used to
        stop the time (i.e. cancel the callback being called), and/or
        to restart the timer.

        It's not necessary to hold a reference to the timer object; a
        ref is held automatically, and discarded when the timer ends or stops.
        """
        timer = self._TimerClass(self, callback, *args, one_shot=True)
        timer.start(delay)
        return timer

    def call_repeated(self, interval, callback, *args):
        """Arrange for a callback to be called repeatedly.

        Returns a timer object (in multi-shot mode) that can be used for
        further control.

        It's not necessary to hold a reference to the timer object; a
        ref is held automatically, and discarded when the timer is
        stopped.
        """
        timer = self._TimerClass(self, callback, *args, one_shot=False)
        timer.start()
        return timer

    def run(self, stop_when_no_canvases=True):
        """Enter the main loop.

        This provides a generic API to start the loop. When building an application (e.g. with Qt)
        its fine to start the loop in the normal way.
        """
        self._stop_when_no_canvases = bool(stop_when_no_canvases)
        self._run()

    def stop(self):
        """Stop the currently running event loop."""
        self._stop()

    def _run(self):
        """For the subclass to implement:

        * Start the event loop.
        * The rest of the loop object must work just fine, also when the loop is
          started in the "normal way" (i.e. this method may not be called).
        """
        raise NotImplementedError()

    def _stop(self):
        """For the subclass to implement:

        * Stop the running event loop.
        * When running in an interactive session, this call should probably be ignored.
        """
        raise NotImplementedError()

    def _call_soon(self, callback, *args):
        """For the subclass to implement:

        * A quick path to have callback called in a next invocation of the event loop.
        * This method is optional: the default implementation just calls ``call_later()`` with a zero delay.
        """
        self.call_later(0, callback, *args)

    def _wgpu_gui_poll(self):
        """For the subclass to implement:

        Some event loops (e.g. asyncio) are just that and dont have a GUI to update.
        Other loops (like Qt) already process events. So this is only intended for
        backends like glfw.
        """
        pass


class AnimationScheduler:
    """
    Some ideas:

    * canvas.add_event_handler("animate", callback)
    * canvas.animate.add_handler(1/30, callback)
    """

    # def iter(self):
    #     # Something like this?
    #     for scheduler in all_schedulers:
    #         scheduler._event_emitter.submit_and_dispatch(event)


class UpdateMode(Enum):
    """The different modes to schedule draws for the canvas."""

    manual = None  #: Draw events are never scheduled. Draws only happen when you ``canvas.force_draw()``, and maybe when the GUI system issues them (e.g. when resizing).
    ondemand = None  #: Draws are only scheduled when ``canvas.request_draw()`` is called when an update is needed. Safes your laptop battery. Honours ``min_fps`` and ``max_fps``.
    continuous = None  #: Continuously schedules draw events, honouring ``max_fps``. Calls to ``canvas.request_draw()`` have no effect.
    fastest = None  #: Continuously schedules draw events as fast as possible. Gives high FPS (and drains your battery).


class Scheduler:
    """Helper class to schedule event processing and drawing."""

    # This class makes the canvas tick. Since we do not own the event-loop, but
    # ride on e.g. Qt, asyncio, wx, JS, or something else, our little "loop" is
    # implemented with a timer.
    #
    # The loop looks a little like this:
    #
    #     ________________      __      ________________      __      rd = request_draw
    #   /   wait           \  / rd \  /   wait           \  / rd \
    #  |                    ||      ||                    ||      |
    # --------------------------------------------------------------------> time
    #  |                    |       |                     |       |
    #  |                    |       draw                  |       draw
    #  schedule             tick                          tick
    #
    # With update modes 'ondemand' and 'manual', the loop ticks at the same rate
    # as on 'continuous' mode, but won't draw every tick:
    #
    #     ________________     ________________      __
    #   /    wait          \  /   wait          \  / rd \
    #  |                    ||                   ||      |
    # --------------------------------------------------------------------> time
    #  |                    |                    |       |
    #  |                    |                    |       draw
    #  schedule             tick                tick
    #
    # A tick is scheduled by calling _schedule_next_tick(). If this method is
    # called when the timer is already running, it has no effect. In the _tick()
    # method, events are processed (including animations). Then, depending on
    # the mode and whether a draw was requested, a new tick is scheduled, or a
    # draw is requested. In the latter case, the timer is not started, but we
    # wait for the canvas to perform a draw. In _draw_drame_and_present() the
    # draw is done, and a new tick is scheduled.
    #
    # The next tick is scheduled when a draw is done, and not earlier, otherwise
    # the drawing may not keep up with the event loop.
    #
    # On desktop canvases the draw usually occurs very soon after it is
    # requested, but on remote frame buffers, it may take a bit longer. To make
    # sure the rendered image reflects the latest state, events are also
    # processed right before doing the draw.
    #
    # When the window is minimized, the draw will not occur until the window is
    # shown again. For the canvas to detect minimized-state, it will need to
    # receive GUI events. This is one of the reasons why the loop object also
    # runs a timer-loop.
    #
    # The drawing itself may take longer than the intended wait time. In that
    # case, it will simply take longer than we hoped and get a lower fps.
    #
    # Note that any extra draws, e.g. via force_draw() or due to window resizes,
    # don't affect the scheduling loop; they are just extra draws.

    def __init__(self, canvas, loop, *, mode="ondemand", min_fps=1, max_fps=30):
        # Objects related to the canvas.
        # We don't keep a ref to the canvas to help gc. This scheduler object can be
        # referenced via a callback in an event loop, but it won't prevent the canvas
        # from being deleted!
        self._canvas_ref = weakref.ref(canvas)
        self._events = canvas._events
        # ... = canvas.get_context() -> No, context creation should be lazy!

        # We need to call_later and process gui events. The loop object abstracts these.
        assert loop is not None
        loop._register_scheduler(self)

        # Scheduling variables
        if mode not in UpdateMode:
            raise ValueError(
                f"Invalid update_mode '{mode}', must be in {set(UpdateMode)}."
            )
        self._mode = mode
        self._min_fps = float(min_fps)
        self._max_fps = float(max_fps)
        self._draw_requested = True  # Start with a draw in ondemand mode

        # Stats
        self._last_draw_time = 0
        self._draw_stats = 0, time.perf_counter()

        # Variables for animation
        self._animation_time = 0
        self._animation_step = 1 / 20

        # Initialise the scheduling loop. Note that the gui may do a first draw
        # earlier, starting the loop, and that's fine.
        self._last_tick_time = -0.1
        self._timer = loop.call_later(0.1, self._tick)

    def _get_canvas(self):
        canvas = self._canvas_ref()
        if canvas is None or canvas.is_closed():
            # Pretty nice, we can send a close event, even if the canvas no longer exists
            self._events._wgpu_close()
            return None
        else:
            return canvas

    def request_draw(self):
        """Request a new draw to be done. Only affects the 'ondemand' mode."""
        # Just set the flag
        self._draw_requested = True

    def _schedule_next_tick(self):
        """Schedule _tick() to be called via our timer."""

        if self._timer.is_running:
            return

        # Determine delay
        if self._mode == "fastest":
            delay = 0
        else:
            delay = 1 / self._max_fps
            delay = 0 if delay < 0 else delay  # 0 means cannot keep up

        # Offset delay for time spent on processing events, etc.
        time_since_tick_start = time.perf_counter() - self._last_tick_time
        delay -= time_since_tick_start
        delay = max(0, delay)

        # Go!
        self._timer.start(delay)

    def _tick(self):
        """Process event and schedule a new draw or tick."""

        self._last_tick_time = time.perf_counter()

        # Get canvas or stop
        if (canvas := self._get_canvas()) is None:
            return

        # Determine what to do next ...

        if self._mode == "fastest":
            # fastest: draw continuously as fast as possible, ignoring fps settings.
            canvas._request_draw()

        elif self._mode == "continuous":
            # continuous: draw continuously, aiming for a steady max framerate.
            canvas._request_draw()

        elif self._mode == "ondemand":
            # ondemand: draw when needed (detected by calls to request_draw).
            # Aim for max_fps when drawing is needed, otherwise min_fps.
            its_time_to_draw = (
                time.perf_counter() - self._last_draw_time > 1 / self._min_fps
            )
            if not self._draw_requested:
                canvas._process_events()  # handlers may request a draw
            if self._draw_requested or its_time_to_draw:
                canvas._request_draw()
            else:
                self._schedule_next_tick()

        elif self._mode == "manual":
            # manual: never draw, except when ... ?
            canvas._process_events()
            self._schedule_next_tick()

        else:
            raise RuntimeError(f"Unexpected scheduling mode: '{self._mode}'")

    def on_draw(self):
        """Called from canvas._draw_frame_and_present()."""

        # It could be that the canvas is closed now. When that happens,
        # we stop here and do not schedule a new iter.
        if (canvas := self._get_canvas()) is None:
            return

        # Update stats
        count, last_time = self._draw_stats
        if time.perf_counter() - last_time > 1.0:
            self._draw_stats = 0, time.perf_counter()
        else:
            self._draw_stats = count + 1, last_time

        # Stats (uncomment to see fps)
        count, last_time = self._draw_stats
        fps = count / (time.perf_counter() - last_time)
        canvas.set_title(f"wgpu {fps:0.1f} fps")

        # Bookkeeping
        self._last_draw_time = time.perf_counter()
        self._draw_requested = False

        # Keep ticking
        self._schedule_next_tick()
