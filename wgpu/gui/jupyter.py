"""
Support for rendering in a Jupyter widget. Provides a widget subclass that
can be used as cell output, or embedded in a ipywidgets gui.
"""

from .offscreen import WgpuOffscreenCanvas

import numpy as np
from jupyter_rfb import RemoteFrameBuffer


class JupyterWgpuCanvas(WgpuOffscreenCanvas, RemoteFrameBuffer):
    """An ipywidgets widget providing a wgpu canvas. Needs the jupyter_rfb library."""

    def __init__(self):
        super().__init__()
        self._pixel_ratio = 1
        self._logical_size = 0, 0
        self._is_closed = False

    # Implementation needed for RemoteFrameBuffer

    def handle_event(self, event):
        event_type = event.get("event_type", "")
        if event_type == "close":
            self._is_closed = True
        elif event_type == "resize":
            self._pixel_ratio = event["pixel_ratio"]
            self._logical_size = event["width"], event["height"]

    def get_frame(self):
        # The _draw_frame_and_present() does the drawing and then calls
        # present_context.present(), which calls our present() method.
        # The resuls is either a numpy array or None, and this matches
        # with what this method is expected to return.
        return self._draw_frame_and_present()

    # Implementation needed for WgpuCanvasBase

    def get_pixel_ratio(self):
        return self._pixel_ratio

    def get_logical_size(self):
        return self._logical_size

    def get_physical_size(self):
        return int(self._logical_size[0] * self._pixel_ratio), int(
            self._logical_size[1] * self._pixel_ratio
        )

    def set_logical_size(self, width, height):
        self.css_width = f"{width}px"
        self.css_height = f"{height}px"

    def close(self):
        RemoteFrameBuffer.close(self)

    def is_closed(self):
        return self._is_closed

    def _request_draw(self):
        RemoteFrameBuffer.request_draw(self)

    # Implementation needed for WgpuOffscreenCanvas

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

    def get_preferred_format(self):
        # Use a format that maps well to PNG: rgba8norm.
        # Use srgb for perseptive color mapping. You probably want to
        # apply this before displaying on screen, but you do not want
        # to duplicate it. When a PNG is shown on screen in the browser
        # it's shown as-is (at least it was when I just tried).
        return "rgba8unorm-srgb"


# Make available under a name that is the same for all gui backends
WgpuCanvas = JupyterWgpuCanvas
