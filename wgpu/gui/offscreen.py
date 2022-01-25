import asyncio

import numpy as np

from ._offscreen import WgpuOffscreenCanvas
from .base import WgpuAutoGui


class WgpuManualOffscreenCanvas(WgpuAutoGui, WgpuOffscreenCanvas):
    """An offscreen canvas intended for manual use. Call the ``.draw()``
    method to perform a draw and get the result.
    """

    def __init__(self, *args, width=640, height=480, pixel_ratio=1, **kwargs):
        super().__init__(*args, **kwargs)
        self._logical_size = width, height
        self._pixel_ratio = pixel_ratio

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
        pass

    def is_closed(self):
        return False

    def _request_draw(self):
        pass

    def present(self, texture_view):
        # This gets called at the end of a draw pass via GPUCanvasContextOffline
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
        return np.frombuffer(data, np.uint8).reshape(size[1], size[0], 4)

    def draw(self):
        """Perform a draw and return the numpy array as a result."""
        return self._draw_frame_and_present()


WgpuCanvas = WgpuManualOffscreenCanvas


def call_later(delay, callback, *args):
    loop = asyncio.get_event_loop_policy().get_event_loop()
    loop.call_later(delay, callback, *args)


async def mainloop_iter():
    pass  # no op


def run():
    """Handle all tasks scheduled with call_later and return."""
    loop = asyncio.get_event_loop_policy().get_event_loop()
    loop.run_until_complete(mainloop_iter())
