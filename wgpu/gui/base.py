import sys
import time

from ._gui_utils import log_exception
from ._events import EventEmitter
from ._loop import WgpuLoop, Scheduler


class WgpuCanvasInterface:
    """The minimal interface to be a valid canvas.

    Any object that implements these methods is a canvas that wgpu can work with.
    The object does not even have to derive from this class.

    In most cases it's more convenient to subclass :class:`WgpuCanvasBase <wgpu.gui.WgpuCanvasBase>`.
    """

    def __init__(self, *args, **kwargs):
        # The args/kwargs are there because we may be mixed with e.g. a Qt widget
        super().__init__(*args, **kwargs)
        self._canvas_context = None

    def get_present_info(self):
        """Get information about the surface to render to.

        It must return a small dict, used by the canvas-context to determine
        how the rendered result should be presented to the canvas. There are
        two possible methods.

        If the ``method`` field is "screen", the context will render directly
        to a surface representing the region on the screen. The dict should
        have a ``window`` field containing the window id. On Linux there should
        also be ``platform`` field to distinguish between "wayland" and "x11",
        and a ``display`` field for the display id. This information is used
        by wgpu to obtain the required surface id.

        When the ``method`` field is "image", the context will render to a
        texture, download the result to RAM, and call ``canvas.present_image()``
        with the image data. Additional info (like format) is passed as kwargs.
        This method enables various types of canvases (including remote ones),
        but note that it has a performance penalty compared to rendering
        directly to the screen.

        The dict can further contain fields ``formats`` and ``alpha_modes`` to
        define the canvas capabilities. For the "image" method, the default
        formats is ``["rgba8unorm-srgb", "rgba8unorm"]``, and the default
        alpha_modes is ``["opaque"]``.
        """
        raise NotImplementedError()

    def get_physical_size(self):
        """Get the physical size of the canvas in integer pixels."""
        raise NotImplementedError()

    def get_context(self, kind="webgpu"):
        """Get the ``GPUCanvasContext`` object corresponding to this canvas.

        The context is used to obtain a texture to render to, and to
        present that texture to the canvas. This class provides a
        default implementation to get the appropriate context.

        The ``kind`` argument is a remnant from the WebGPU spec and
        must always be "webgpu".
        """
        # Note that this function is analog to HtmlCanvas.getContext(), except
        # here the only valid arg is 'webgpu', which is also made the default.
        assert kind == "webgpu"
        if self._canvas_context is None:
            backend_module = sys.modules["wgpu"].gpu.__module__
            if backend_module == "wgpu._classes":
                raise RuntimeError(
                    "A backend must be selected (e.g. with request_adapter()) before canvas.get_context() can be called."
                )
            CanvasContext = sys.modules[backend_module].GPUCanvasContext  # noqa: N806
            self._canvas_context = CanvasContext(self)
        return self._canvas_context

    def present_image(self, image, **kwargs):
        """Consume the final rendered image.

        This is called when using the "image" method, see ``get_present_info()``.
        Canvases that don't support offscreen rendering don't need to implement
        this method.
        """
        raise NotImplementedError()


class WgpuCanvasBase(WgpuCanvasInterface):
    """A convenient base canvas class.

    This class provides a uniform API and implements common
    functionality, to increase consistency and reduce code duplication.
    It is convenient (but not strictly necessary) for canvas classes
    to inherit from this class (but all builtin canvases do).

    This class provides an API for scheduling draws (``request_draw()``)
    and implements a mechanism to call the provided draw function
    (``draw_frame()``) and then present the result to the canvas.

    This class also implements draw rate limiting, which can be set
    with the ``max_fps`` attribute (default 30). For benchmarks you may
    also want to set ``vsync`` to False.
    """

    def __init__(
        self, *args, max_fps=30, vsync=True, present_method=None, ticking=True, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._min_fps = float(1.0)
        self._max_fps = float(max_fps)
        self._vsync = bool(vsync)
        present_method  # noqa - We just catch the arg here in case a backend does implement support it

        self._draw_frame = lambda: None
        self._events = EventEmitter()
        # self._scheduler = Scheduler(self)

        self._draw_requested = True
        self._schedule_time = 0
        self._last_draw_time = 0
        self._draw_stats = 0, time.perf_counter()
        self._mode = "continuous"

        self._a_tick_is_scheduled = False

        self._animation_time = 0
        self._animation_step = 1 / 20

        if ticking:
            self._schedule_tick()

    def __del__(self):
        # On delete, we call the custom close method.
        try:
            self.close()
        except Exception:
            pass
        # Since this is sometimes used in a multiple inheritance, the
        # superclass may (or may not) have a __del__ method.
        try:
            super().__del__()
        except Exception:
            pass

    @property
    def events(self):
        return self._events

    @property
    def scheduler(self):
        return self._scheduler

    @property
    def loop(self):
        return self._get_loop()

    def request_draw(self, draw_function=None):
        """Schedule a new draw event.

        This function does not perform a draw directly, but schedules
        a draw event at a suitable moment in time. In the draw event
        the draw function is called, and the resulting rendered image
        is presented to screen.

        Arguments:
            draw_function (callable or None): The function to set as the new draw
                function. If not given or None, the last set draw function is used.

        """
        if draw_function is not None:
            self._draw_frame = draw_function

        # We don't call self._request_draw() directly but let the scheduler do that based on the policy
        # self._scheduler.request_draw()
        # todo: maybe have set_draw_function() separately
        # todo: maybe requesting a new draw can be done by setting a field in an event?
        # todo: can we invoke the draw function via a draw event?

        # We can assume that this function is called when we flush events.
        # So we can also maybe replace this by letting downstream code set a flag on the event object.
        # In any case, we only really have to do something in ondemand mode; in other modes we draw regardless.
        self._draw_requested = True

    def force_draw(self):
        self._force_draw()

    def _schedule_tick(self):
        # This method makes the canvas tick. Since we do not own the event-loop,
        # but ride on e.g. Qt, asyncio, wx, JS, or something else, our little
        # "loop" is implemented with call_later calls. It's crucial that the
        # loop stays clean and does not 'duplicate', e.g. by an extra draw being
        # done behind our back, otherwise the fps might double (but be
        # irregular). Taking care of this is surprising tricky.
        #
        #     ________________      __      ________________      __     ________________      __
        #   /    call_later    \  / rd \  /   call_later     \  / rd \  /                 \  /    \
        #  |                    ||      ||                    ||      ||                   ||      |
        # ---------------------------------------------------------------------------------------------> time
        #  |                    |       |                     |       |
        #  |                    |       draw                  |       draw
        #  schedule_tick        tick                          tick
        #
        #
        # In between the calls to _schedule_tick() and tick(), a new
        # tick cannot be invoked. In tick() the _request_draw() method is
        # called that asks the GUI to schedule a draw. The first thing that the
        # draw() method does, is schedule a new draw. In effect, any extra draws
        # that are performed do not affect the ticking itself.
        #
        #     ________________     ________________      __     ________________      __
        #   /    call_later    \  /   call_later    \  / rd \  /                 \  /    \
        #  |                    ||                   ||      ||                   ||      |
        # ---------------------------------------------------------------------------------------------> time
        #  |                    |                    |       |
        #  |                    |                    |       draw
        #  schedule             tick                 tick

        # This method gets called right before/after the draw is performed, from
        # _draw_frame_and_present(). In here, we make scheduler that a new draw
        # is done (by the undelying GUI), so that _draw_frame_and_present() gets
        # called again. We cannot implement a loop-thingy that occasionally
        # schedules a draw event, because if the drawing cannot keep up, the
        # requests pile up and got out of sync.

        # Prevent recursion. This is important, otherwise an extra call results in duplicate drawing.
        if self._a_tick_is_scheduled:
            return
        self._a_tick_is_scheduled = True
        self._schedule_time = time.perf_counter()

        def tick():
            # Determine whether to request a draw or just schedule a new tick
            if self._mode == "manual":
                # manual: never draw, except when ..... ?
                self._flush_events()
                request_a_draw = False
            elif self._mode == "ondemand":
                # ondemand: draw when needed (detected by calls to request_draw). Aim for max_fps when drawing is needed, otherwise min_fps.
                self._flush_events()  # may set _draw_requested
                its_draw_time = (
                    time.perf_counter() - self._last_draw_time > 1 / self._min_fps
                )
                request_a_draw = self._draw_requested or its_draw_time
            elif self._mode == "continuous":
                # continuous: draw continuously, aiming for a steady max framerate.
                request_a_draw = True
            else:
                # fastest: draw continuously as fast as possible, ignoring fps settings.
                request_a_draw = True

            # Request a draw, or flush events and schedule again.
            self._a_tick_is_scheduled = False
            if request_a_draw:
                self._request_draw()
            else:
                self._schedule_tick()

        loop = self._get_loop()
        if self._mode == "fastest":
            # Draw continuously as fast as possible, ignoring fps settings.
            loop.call_soon(tick)
        else:
            # Schedule a new tick
            delay = 1 / self._max_fps
            delay = 0 if delay < 0 else delay  # 0 means cannot keep up
            loop.call_later(delay, tick)

    def _process_input(self):
        """This should process all GUI events.

        In some GUI systems, like Qt, events are already processed because the
        Qt event loop is running, so this can be a no-op. In other cases, like
        glfw, this hook allows glfw to do a tick.
        """
        raise NotImplementedError()

    def _flush_events(self):
        # Get events from the GUI into our event mechanism.
        self._get_loop().poll()  # todo: maybe self._process_gui_events()?

        # Flush our events, so downstream code can update stuff.
        # Maybe that downstream code request a new draw.
        self.events.flush()

        # Schedule events until the lag is gone
        step = self._animation_step
        self._animation_time = self._animation_time or time.perf_counter()  # start now
        animation_iters = 0
        while self._animation_time > time.perf_counter() - step:
            self._animation_time += step
            self.events.submit({"event_type": "animate", "step": step})
            # Do the animations. This costs time.
            self.events.flush()
            # Abort when we cannot keep up
            # todo: test this
            animation_iters += 1
            if animation_iters > 20:
                n = (time.perf_counter() - self._animation_time) // step
                self._animation_time += step * n
                self.events.submit(
                    {"event_type": "animate", "step": step * n, "catch_up": n}
                )

    # todo: was _draw_frame_and_present
    def _tick_draw(self):
        """Draw the frame and present the result.

        Errors are logged to the "wgpu" logger. Should be called by the
        subclass at an appropriate time.
        """
        # This method is called from the GUI layer. It can be called from a "draw event" that we requested, or as part of a forced draw.
        # So this call must to the complete tick.

        self._draw_requested = False
        self._schedule_tick()

        self._flush_events()

        # It could be that the canvas is closed now. When that happens,
        # we stop here and do not schedule a new iter.
        if self.is_closed():
            return

        self.events.submit({"event_type": "before_draw"})
        self.events.flush()

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
        self.set_title(f"wgpu {fps:0.1f} fps")

        # Perform the user-defined drawing code. When this errors,
        # we should report the error and then continue, otherwise we crash.
        # Returns the result of the context's present() call or None.
        # todo: maybe move to scheduler
        with log_exception("Draw error"):
            self._draw_frame()
        with log_exception("Present error"):
            if self._canvas_context:
                time.sleep(0.01)
                return self._canvas_context.present()

    # Methods that must be overloaded to provided a common API for downstream libraries and end-users

    def _get_loop(self):
        """Must return the global loop instance."""
        raise NotImplementedError()

    def _request_draw(self):
        """Like requestAnimationFrame in JS. Must schedule a call to self._scheduler.draw() ???"""
        raise NotImplementedError()

    def _force_draw(self):
        """Perform a draw right now."""
        raise NotImplementedError()

    def get_pixel_ratio(self):
        """Get the float ratio between logical and physical pixels."""
        raise NotImplementedError()

    def get_logical_size(self):
        """Get the logical size in float pixels."""
        raise NotImplementedError()

    def get_physical_size(self):
        """Get the physical size in integer pixels."""
        raise NotImplementedError()

    def set_logical_size(self, width, height):
        """Set the window size (in logical pixels)."""
        raise NotImplementedError()

    def set_title(self, title):
        """Set the window title."""
        raise NotImplementedError()

    def close(self):
        """Close the window."""
        pass

    def is_closed(self):
        """Get whether the window is closed."""
        raise NotImplementedError()
