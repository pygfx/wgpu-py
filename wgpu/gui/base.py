import sys

from ._events import EventEmitter
from ._loop import Scheduler, WgpuLoop  # noqa: F401


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
        self,
        *args,
        max_fps=30,
        vsync=True,
        present_method=None,
        use_scheduler=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._vsync = bool(vsync)
        present_method  # noqa - We just catch the arg here in case a backend does implement support it

        self._draw_frame = lambda: None

        self._events = EventEmitter()

        self._scheduler = None
        if use_scheduler:
            self._scheduler = Scheduler(self, max_fps=max_fps)

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

    add_event_handler.__doc__ = EventEmitter.add_handler.__doc__
    remove_event_handler.__doc__ = EventEmitter.remove_handler.__doc__

    # === Scheduling

    @property
    def loop(self):
        return self._get_loop()

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
        if self._scheduler is None:
            return
        if draw_function is not None:
            self._draw_frame = draw_function
        self._scheduler.request_draw()

        # todo: maybe requesting a new draw can be done by setting a field in an event?
        # todo: can just make the draw_function a handler for the draw event?
        # -> Note that the draw func is likely to hold a ref to the canvas. By storing it
        #   here, the circular ref can be broken. This fails if we'd store _draw_frame on the
        #   scheduler! So with a draw event, we should provide the context and more info so
        #   that a draw funcion does not need the canvas object.

    def force_draw(self):
        self._force_draw()

    def _draw_frame_and_present(self):
        """Draw the frame and present the result.

        Errors are logged to the "wgpu" logger. Should be called by the
        subclass at an appropriate time.
        """
        # This method is called from the GUI layer. It can be called from a "draw event" that we requested, or as part of a forced draw.
        # So this call must to the complete tick.
        if self._scheduler is not None:
            self._scheduler.draw_tick()

    # === Canvas management methods

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
