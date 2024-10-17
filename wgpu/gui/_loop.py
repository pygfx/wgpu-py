import time
import weakref

from ._gui_utils import log_exception

# Note: technically, we could have a global loop proxy object that defers to any of the other loops.
# That would e.g. allow using glfw with qt together. Probably to too weird use-case for the added complexity.


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

    def _wgpu_gui_poll(self):
        """Poll the underlying GUI toolkit for window events.

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

    * canvas.add_event_handler("animate", callback)
    * canvas.animate.add_handler(1/30, callback)
    """

    # def iter(self):
    #     # Something like this?
    #     for scheduler in all_schedulers:
    #         scheduler._event_emitter.submit_and_dispatch(event)


class Scheduler:
    """Helper class to schedule event processing and drawing."""

    # This class makes the canvas tick. Since we do not own the event-loop, but
    # ride on e.g. Qt, asyncio, wx, JS, or something else, our little "loop" is
    # implemented with call_later calls. It's crucial that the loop stays clean
    # and does not 'duplicate', e.g. by an extra draw being done behind our
    # back, otherwise the fps might double (but be irregular). Taking care of
    # this is surprising tricky.
    #
    # The loop looks a little like this:
    #
    #     ________________      __      ________________      __
    #   /    call_later    \  / rd \  /   call_later     \  / rd \
    #  |                    ||      ||                    ||      |
    # --------------------------------------------------------------------> time
    #  |                    |       |                     |       |
    #  |                    |       draw_tick             |       draw_tick
    #  schedule             pseuso_tick                   pseudo_tick
    #
    #
    # While the loop is waiting - via call_later, in between the calls to
    # _schedule_tick() and pseudo_tick() - a new tick cannot be scheduled. In
    # pseudo_tick() the _request_draw() method is called that asks the GUI to
    # schedule a draw. This happens in a later event-loop iteration, an can
    # happen (nearly) directly, or somewhat later. The first thing that the
    # draw_tick() method does, is schedule a new draw. Any extra draws that are
    # performed still call _schedule_tick(), but this has no effect.
    #
    # With update modes 'ondemand' and 'manual', the loop ticks at the same rate
    # as on 'continuous' mode, but won't draw every tick. The event_tick() is
    # then called instead, so that events handlers and animations stay active,
    # from which a new draw may be requested.
    #
    #     ________________     ________________      __
    #   /    call_later    \  /   call_later    \  / rd \
    #  |                    ||                   ||      |
    # --------------------------------------------------------------------> time
    #  |                    |                    |       |
    #  |                    |                    |       draw_tick
    #  schedule             pseuso_tick          pseuso_tick
    #                     + event_tick

    def __init__(self, canvas, loop, *, mode="ondemand", min_fps=1, max_fps=30):
        # Objects related to the canvas.
        # We don't keep a ref to the canvas to help gc. This scheduler object can be
        # referenced via a callback in an event loop, but it won't prevent the canvas
        # from being deleted!
        self._canvas_ref = weakref.ref(canvas)
        self._events = canvas._events
        # ... = canvas.get_context() -> No, context creation should be lazy!

        # We need to call_later and process gui events. The loop object abstracts these.
        self._loop = loop
        assert loop is not None

        # Lock the scheduling while its waiting
        self._waiting_lock = False

        # Scheduling variables
        self._mode = mode
        self._min_fps = float(min_fps)
        self._max_fps = float(max_fps)
        self._draw_requested = True

        # Stats
        self._last_draw_time = 0
        self._draw_stats = 0, time.perf_counter()

        # Variables for animation
        self._animation_time = 0
        self._animation_step = 1 / 20

        # Start by doing the first scheduling.
        # Note that the gui may do a first draw earlier, starting the loop, and that's fine.
        self._loop.call_later(0.1, self._schedule_next_tick)

    def _get_canvas(self):
        canvas = self._canvas_ref()
        if not (canvas is None or canvas.is_closed()):
            return canvas

    def request_draw(self):
        """Request a new draw to be done. Only affects the 'ondemand' mode."""
        # Just set the flag
        self._draw_requested = True

    def _schedule_next_tick(self):
        # Scheduling flow:
        #
        # * _schedule_next_tick():
        #   * determine delay
        #   * use call_later() to have pseudo_tick() called
        #
        # * pseudo_tick():
        #   * decide whether to request a draw
        #     * a draw is requested:
        #       * the GUI will call canvas._draw_frame_and_present()
        #       * wich calls draw_tick()
        #     * A draw is not requested:
        #       * call event_tick()
        #       * call _schedule_next_tick()
        #
        # * event_tick():
        #   * let GUI process events
        #   * flush events
        #   * run animations
        #
        # * draw_tick():
        #   * calls _schedule_next_tick()
        #   * calls event_tick()
        #   * draw!

        # Notes:
        #
        # * New ticks must be scheduled from the draw_tick, otherwise
        #   new draws may get scheduled faster than it can keep up.
        # * It's crucial to not have two cycles running at the same time.
        # * We must assume that the GUI can do extra draws (i.e. draw_tick gets called) any time, e.g. when resizing.

        # Flag that allows this method to be called at any time, without introducing an extra "loop".
        if self._waiting_lock:
            return
        self._waiting_lock = True

        # Determine delay
        if self._mode == "fastest":
            delay = 0
        else:
            delay = 1 / self._max_fps
            delay = 0 if delay < 0 else delay  # 0 means cannot keep up

        def pseudo_tick():
            # Since this resets the waiting lock, we really want to avoid accidentally
            # calling this function. That's why we define it locally.

            # Enable scheduling again
            self._waiting_lock = False

            # Get canvas or stop
            if (canvas := self._get_canvas()) is None:
                return

            if self._mode == "fastest":
                # fastest: draw continuously as fast as possible, ignoring fps settings.
                canvas._request_draw()

            elif self._mode == "continuous":
                # continuous: draw continuously, aiming for a steady max framerate.
                canvas._request_draw()

            elif self._mode == "ondemand":
                # ondemand: draw when needed (detected by calls to request_draw).
                # Aim for max_fps when drawing is needed, otherwise min_fps.
                self.event_tick()  # may set _draw_requested
                its_draw_time = (
                    time.perf_counter() - self._last_draw_time > 1 / self._min_fps
                )
                if self._draw_requested or its_draw_time:
                    canvas._request_draw()
                else:
                    self._schedule_next_tick()

            elif self._mode == "manual":
                # manual: never draw, except when ... ?
                self.event_tick()
                self._schedule_next_tick()

            else:
                raise RuntimeError(f"Unexpected scheduling mode: '{self._mode}'")

        self._loop.call_later(delay, pseudo_tick)

    def event_tick(self):
        """A lightweight tick that processes evets and animations."""

        # Get events from the GUI into our event mechanism.
        self._loop._wgpu_gui_poll()

        # Flush our events, so downstream code can update stuff.
        # Maybe that downstream code request a new draw.
        self._events.flush()

        # Schedule animation events until the lag is gone
        step = self._animation_step
        self._animation_time = self._animation_time or time.perf_counter()  # start now
        animation_iters = 0
        while self._animation_time > time.perf_counter() - step:
            self._animation_time += step
            self._events.submit({"event_type": "animate", "step": step, "catch_up": 0})
            # Do the animations. This costs time.
            self._events.flush()
            # Abort when we cannot keep up
            # todo: test this
            animation_iters += 1
            if animation_iters > 20:
                n = (time.perf_counter() - self._animation_time) // step
                self._animation_time += step * n
                self._events.submit(
                    {"event_type": "animate", "step": step * n, "catch_up": n}
                )

    def draw_tick(self):
        """Perform a full tick: processing events, animations, drawing, and presenting."""

        # Events and animations
        self.event_tick()

        # It could be that the canvas is closed now. When that happens,
        # we stop here and do not schedule a new iter.
        if (canvas := self._get_canvas()) is None:
            return

        # Keep ticking
        self._draw_requested = False
        self._schedule_next_tick()

        # Special event for drawing
        self._events.submit({"event_type": "before_draw"})
        self._events.flush()

        # Schedule a new draw right before doing the draw. Important that it happens *after* processing events.
        self._last_draw_time = time.perf_counter()

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

        # Perform the user-defined drawing code. When this errors,
        # we should report the error and then continue, otherwise we crash.
        with log_exception("Draw error"):
            canvas._draw_frame()
        with log_exception("Present error"):
            # Note: we use canvas._canvas_context, so that if the draw_frame is a stub we also dont trigger creating a context.
            # Note: if vsync is used, this call may wait a little (happens down at the level of the driver or OS)
            context = canvas._canvas_context
            if context:
                context.present()
