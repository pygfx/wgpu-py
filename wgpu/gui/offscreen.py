import time

from ._offscreen import WgpuOffscreenCanvas
from .base import WgpuAutoGui


class WgpuManualOffscreenCanvas(WgpuAutoGui, WgpuOffscreenCanvas):
    """An offscreen canvas intended for manual use.

    Call the ``.draw()`` method to perform a draw and get the result.
    """

    def __init__(self, *args, size=None, pixel_ratio=1, title=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._logical_size = (float(size[0]), float(size[1])) if size else (640, 480)
        self._pixel_ratio = pixel_ratio
        self._title = title
        self._closed = False

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

    def close(self):
        self._closed = True

    def is_closed(self):
        return self._closed

    def _request_draw(self):
        # Deliberately a no-op, because people use .draw() instead.
        pass

    def present(self, texture_view):
        # This gets called at the end of a draw pass via _offscreen.GPUCanvasContext
        device = texture_view._device
        size = texture_view.size
        bytes_per_pixel = 4
        data = device.queue.read_texture(
            {
                "texture": texture_view.texture,
                "mip_level": 0,
                "origin": (0, 0, 0),
            },
            {
                "offset": 0,
                "bytes_per_row": bytes_per_pixel * size[0],
                "rows_per_image": size[1],
            },
            size,
        )

        # Return as memory object to avoid numpy dependency
        # Equivalent: np.frombuffer(data, np.uint8).reshape(size[1], size[0], 4)
        return data.cast("B", (size[1], size[0], 4))

    def draw(self):
        """Perform a draw and return the resulting array as an NxMx4 memoryview object.
        This object can be converted to a numpy array (without copying data)
        using ``np.asarray(arr)``.
        """
        return self._draw_frame_and_present()


WgpuCanvas = WgpuManualOffscreenCanvas


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

_pending_calls = []


def call_later(delay, callback, *args):
    # Note that this module never calls call_later() itself; request_draw() is a no-op.
    etime = time.time() + delay
    _pending_calls.append((etime, callback, args))


def run():
    # Process pending calls
    for etime, callback, args in _pending_calls.copy():
        if time.time() >= etime:
            callback(*args)

    # Clear any leftover scheduled calls, to avoid lingering refs.
    _pending_calls.clear()
