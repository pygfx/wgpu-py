import sys

from ._events import EventEmitter, WgpuEventType  # noqa: F401
from ._loop import Scheduler, WgpuLoop, WgpuTimer  # noqa: F401
from ._gui_utils import log_exception


class WgpuCanvasInterface:
    """The minimal interface to be a valid canvas.

    Any object that implements these methods is a canvas that wgpu can work with.
    The object does not even have to derive from this class.

    In most cases it's more convenient to subclass :class:`WgpuCanvasBase <wgpu.gui.WgpuCanvasBase>`.
    """

    _canvas_context = None  # set in get_context()

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
    """The base canvas class.

    This class provides a uniform canvas API so render systems can be use
    code that is portable accross multiple GUI libraries and canvas targets.

    Arguments:
        update_mode (WgpuEventType): The mode for scheduling draws and events. Default 'ondemand'.
        min_fps (float): A minimal frames-per-second to use when the ``update_mode`` is 'ondemand'.
            The default is 1: even without draws requested, it still draws every second.
        max_fps (float): A maximal frames-per-second to use when the ``update_mode`` is 'ondemand' or 'continuous'.
            The default is 30, which is usually enough.
        vsync (bool): Whether to sync the draw with the monitor update.  Helps
            against screen tearing, but can reduce fps. Default True.
        present_method (str | None): The method to present the rendered image.
            Can be set to 'screen' or 'image'. Default None (auto-select).

    """

    def __init__(
        self,
        *args,
        update_mode="ondemand",
        min_fps=1.0,
        max_fps=30.0,
        vsync=True,
        present_method=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._vsync = bool(vsync)
        present_method  # noqa - We just catch the arg here in case a backend does implement it

        # Canvas
        self.__raw_title = ""
        self.__title_kwargs = {
            "fps": "?",
            "backend": self.__class__.__name__,
        }

        self.__is_drawing = False
        self._events = EventEmitter()
        self._scheduler = None
        loop = self._get_loop()
        if loop:
            self._scheduler = Scheduler(
                self, loop, min_fps=min_fps, max_fps=max_fps, mode=update_mode
            )

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

    # === Events

    def add_event_handler(self, *args, **kwargs):
        return self._events.add_handler(*args, **kwargs)

    def remove_event_handler(self, *args, **kwargs):
        return self._events.remove_handler(*args, **kwargs)

    def submit_event(self, event):
        # Not strictly necessary for normal use-cases, but this allows
        # the ._event to be an implementation detail to subclasses, and it
        # allows users to e.g. emulate events in tests.
        return self._events.submit(event)

    add_event_handler.__doc__ = EventEmitter.add_handler.__doc__
    remove_event_handler.__doc__ = EventEmitter.remove_handler.__doc__
    submit_event.__doc__ = EventEmitter.submit.__doc__

    def _process_events(self):
        """Process events and animations. Called from the scheduler."""

        # We don't want this to be called too often, because we want the
        # accumulative events to accumulate. Once per draw, and at max_fps
        # when there are no draws (in ondemand and manual mode).

        # Get events from the GUI into our event mechanism.
        loop = self._get_loop()
        if loop:
            loop._wgpu_gui_poll()

        # Flush our events, so downstream code can update stuff.
        # Maybe that downstream code request a new draw.
        self._events.flush()

        # TODO: implement later (this is a start but is not tested)
        # Schedule animation events until the lag is gone
        # step = self._animation_step
        # self._animation_time = self._animation_time or time.perf_counter()  # start now
        # animation_iters = 0
        # while self._animation_time > time.perf_counter() - step:
        #     self._animation_time += step
        #     self._events.submit({"event_type": "animate", "step": step, "catch_up": 0})
        #     # Do the animations. This costs time.
        #     self._events.flush()
        #     # Abort when we cannot keep up
        #     # todo: test this
        #     animation_iters += 1
        #     if animation_iters > 20:
        #         n = (time.perf_counter() - self._animation_time) // step
        #         self._animation_time += step * n
        #         self._events.submit(
        #             {"event_type": "animate", "step": step * n, "catch_up": n}
        #         )

    # === Scheduling and drawing

    def _draw_frame(self):
        """The method to call to draw a frame.

        Cen be overriden by subclassing, or by passing a callable to request_draw().
        """
        pass

    def request_draw(self, draw_function=None):
        """Schedule a new draw event.

        This function does not perform a draw directly, but schedules a draw at
        a suitable moment in time. At that time the draw function is called, and
        the resulting rendered image is presented to screen.

        Only affects drawing with schedule-mode 'ondemand'.

        Arguments:
            draw_function (callable or None): The function to set as the new draw
                function. If not given or None, the last set draw function is used.

        """
        if draw_function is not None:
            self._draw_frame = draw_function
        if self._scheduler is not None:
            self._scheduler.request_draw()

        # -> Note that the draw func is likely to hold a ref to the canvas. By
        #   storing it here, the gc can detect this case, and its fine. However,
        #   this fails if we'd store _draw_frame on the scheduler!

    def force_draw(self):
        """Perform a draw right now.

        In most cases you want to use ``request_draw()``. If you find yourself using
        this, consider using a timer. Nevertheless, sometimes you just want to force
        a draw right now.
        """
        if self.__is_drawing:
            raise RuntimeError("Cannot force a draw while drawing.")
        self._force_draw()

    def _draw_frame_and_present(self):
        """Draw the frame and present the result.

        Errors are logged to the "wgpu" logger. Should be called by the
        subclass at its draw event.
        """

        # Re-entrent drawing is problematic. Let's actively prevent it.
        if self.__is_drawing:
            return
        self.__is_drawing = True

        try:
            # This method is called from the GUI layer. It can be called from a
            # "draw event" that we requested, or as part of a forced draw.

            # Cannot draw to a closed canvas.
            if self.is_closed():
                return

            # Process special events
            # Note that we must not process normal events here, since these can do stuff
            # with the canvas (resize/close/etc) and most GUI systems don't like that.
            self._events.emit({"event_type": "before_draw"})

            # Notify the scheduler
            if self._scheduler is not None:
                fps = self._scheduler.on_draw()

                # Maybe update title
                if fps is not None:
                    self.__title_kwargs["fps"] = f"{fps:0.1f}"
                    if "$fps" in self.__raw_title:
                        self.set_title(self.__raw_title)

            # Perform the user-defined drawing code. When this errors,
            # we should report the error and then continue, otherwise we crash.
            with log_exception("Draw error"):
                self._draw_frame()
            with log_exception("Present error"):
                # Note: we use canvas._canvas_context, so that if the draw_frame is a stub we also dont trigger creating a context.
                # Note: if vsync is used, this call may wait a little (happens down at the level of the driver or OS)
                context = self._canvas_context
                if context:
                    context.present()

        finally:
            self.__is_drawing = False

    def _get_loop(self):
        """For the subclass to implement:

        Must return the global loop instance (WgpuLoop) for the canvas subclass,
        or None for a canvas without scheduled draws.
        """
        return None

    def _request_draw(self):
        """For the subclass to implement:

        Request the GUI layer to perform a draw. Like requestAnimationFrame in
        JS. The draw must be performed by calling _draw_frame_and_present().
        It's the responsibility for the canvas subclass to make sure that a draw
        is made as soon as possible.

        Canvases that have a limit on how fast they can 'consume' frames, like
        remote frame buffers, do good to call self._process_events() when the
        draw had to wait a little. That way the user interaction will lag as
        little as possible.

        The default implementation does nothing, which is equivalent to waiting
        for a forced draw or a draw invoked by the GUI system.
        """
        pass

    def _force_draw(self):
        """For the subclass to implement:

        Perform a synchronous draw. When it returns, the draw must have been done.
        The default implementation just calls _draw_frame_and_present().
        """
        self._draw_frame_and_present()

    # === Primary canvas management methods

    # todo: we require subclasses to implement public methods, while everywhere else the implementable-methods are private.

    def get_physical_size(self):
        """Get the physical size in integer pixels."""
        raise NotImplementedError()

    def get_logical_size(self):
        """Get the logical size in float pixels."""
        raise NotImplementedError()

    def get_pixel_ratio(self):
        """Get the float ratio between logical and physical pixels."""
        raise NotImplementedError()

    def close(self):
        """Close the window."""
        pass

    def is_closed(self):
        """Get whether the window is closed."""
        return False

    # === Secondary canvas management methods

    # These methods provide extra control over the canvas. Subclasses should
    # implement the methods they can, but these features are likely not critical.

    def set_logical_size(self, width, height):
        """Set the window size (in logical pixels)."""
        pass

    def set_title(self, title):
        """Set the window title."""
        self.__raw_title = title
        for k, v in self.__title_kwargs.items():
            title = title.replace("$" + k, v)
        self._set_title(title)

    def _set_title(self, title):
        pass


def pop_kwargs_for_base_canvas(kwargs_dict):
    """Convenience functions for wrapper canvases like in Qt and wx."""
    code = WgpuCanvasBase.__init__.__code__
    base_kwarg_names = code.co_varnames[: code.co_argcount + code.co_kwonlyargcount]
    d = {}
    for key in base_kwarg_names:
        if key in kwargs_dict:
            d[key] = kwargs_dict.pop(key)
    return d
