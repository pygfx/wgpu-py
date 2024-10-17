import time

from .base import WgpuCanvasBase, WgpuLoop


class WgpuManualOffscreenCanvas(WgpuCanvasBase):
    """An offscreen canvas intended for manual use.

    Call the ``.draw()`` method to perform a draw and get the result.
    """

    def __init__(self, *args, size=None, pixel_ratio=1, title=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._logical_size = (float(size[0]), float(size[1])) if size else (640, 480)
        self._pixel_ratio = pixel_ratio
        self._title = title
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

    def set_title(self, title):
        pass

    def close(self):
        self._closed = True

    def is_closed(self):
        return self._closed

    def _get_loop(self):
        return loop

    def _request_draw(self):
        # Deliberately a no-op, because people use .draw() instead.
        pass

    def _force_draw(self):
        self._draw_frame_and_present()

    def draw(self):
        """Perform a draw and get the resulting image.

        The image array is returned as an NxMx4 memoryview object.
        This object can be converted to a numpy array (without copying data)
        using ``np.asarray(arr)``.
        """
        self._force_draw()
        return self._last_image


WgpuCanvas = WgpuManualOffscreenCanvas


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

    def __init__(self):
        super().__init__()
        self._pending_calls = []

    def call_later(self, delay, callback, *args):
        # Note that this module never calls call_later() itself; request_draw() is a no-op.
        etime = time.time() + delay
        self._pending_calls.append((etime, callback, args))

    def run(self):
        # Process pending calls
        for etime, callback, args in self._pending_calls.copy():
            if time.time() >= etime:
                callback(*args)

        # Clear any leftover scheduled calls, to avoid lingering refs.
        self._pending_calls.clear()

    def stop(self):
        pass


loop = StubLoop()
