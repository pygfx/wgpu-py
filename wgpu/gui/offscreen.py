from .base import WgpuCanvasBase, WgpuLoop, WgpuTimer


class WgpuManualOffscreenCanvas(WgpuCanvasBase):
    """An offscreen canvas intended for manual use.

    Call the ``.draw()`` method to perform a draw and get the result.
    """

    def __init__(self, *args, size=None, pixel_ratio=1, title=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._logical_size = (float(size[0]), float(size[1])) if size else (640, 480)
        self._pixel_ratio = pixel_ratio
        self._closed = False
        self._last_image = None

    def get_present_info(self):
        return {
            "method": "image",
            "formats": ["rgba8unorm-srgb", "rgba8unorm"],
        }

    def present_image(self, image, **kwargs):
        self._last_image = image

    def get_pixel_ratio(self):
        return self._pixel_ratio

    def get_logical_size(self):
        return self._logical_size

    def get_physical_size(self):
        return int(self._logical_size[0] * self._pixel_ratio), int(
            self._logical_size[1] * self._pixel_ratio
        )

    def set_logical_size(self, width, height):
        self._logical_size = width, height

    def _set_title(self, title):
        pass

    def close(self):
        self._closed = True

    def is_closed(self):
        return self._closed

    def _get_loop(self):
        return None  # No scheduling

    def _request_draw(self):
        # Ok, cool, the scheduler want a draw. But we only draw when the user
        # calls draw(), so that's how this canvas ticks.
        pass

    def _force_draw(self):
        self._draw_frame_and_present()

    def draw(self):
        """Perform a draw and get the resulting image.

        The image array is returned as an NxMx4 memoryview object.
        This object can be converted to a numpy array (without copying data)
        using ``np.asarray(arr)``.
        """
        loop._process_timers()  # Little trick to keep the event loop going
        self._draw_frame_and_present()
        return self._last_image


WgpuCanvas = WgpuManualOffscreenCanvas


class StubWgpuTimer(WgpuTimer):
    def _start(self):
        pass

    def _stop(self):
        pass


class StubLoop(WgpuLoop):
    # If we consider the use-cases for using this offscreen canvas:
    #
    # * Using wgpu.gui.auto in test-mode: in this case run() should not hang,
    #   and call_later should not cause lingering refs.
    # * Using the offscreen canvas directly, in a script: in this case you
    #   do not have/want an event system.
    # * Using the offscreen canvas in an evented app. In that case you already
    #   have an app with a specific event-loop (it might be PySide6 or
    #   something else entirely).
    #
    # In summary, we provide a call_later() and run() that behave pretty
    # well for the first case.

    _TimerClass = StubWgpuTimer  # subclases must set this

    def _process_timers(self):
        # Running this loop processes any timers
        for timer in list(WgpuTimer._running_timers):
            if timer.time_left <= 0:
                timer._tick()

    def _run(self):
        self._process_timers()

    def _stop(self):
        pass


loop = StubLoop()
