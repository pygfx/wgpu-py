import time

from .. import classes, flags
from .base import WgpuCanvasBase, WgpuAutoGui


class GPUCanvasContext(classes.GPUCanvasContext):
    """GPUCanvasContext subclass for rendering to an offscreen texture."""

    # In this context implementation, we keep a ref to the texture, to keep
    # it alive until at least until present() is called, and to be able to
    # pass it to the canvas' present() method. Thereafter, the texture
    # reference is removed. If there are no more references to it, it will
    # be cleaned up. But if the offscreen canvas uses it for something,
    # it'll simply stay alive longer.

    def __init__(self, canvas):
        super().__init__(canvas)
        self._config = None
        self._texture = None

    def configure(
        self,
        *,
        device,
        format,
        usage=flags.TextureUsage.RENDER_ATTACHMENT | flags.TextureUsage.COPY_SRC,
        view_formats=[],
        color_space="srgb",
        alpha_mode="opaque"
    ):
        if format is None:
            format = self.get_preferred_format(device.adapter)
        self._config = {
            "device": device,
            "format": format,
            "usage": usage,
            "width": 0,
            "height": 0,
            # "view_formats": xx,
            # "color_space": xx,
            # "alpha_mode": xx,
        }

    def unconfigure(self):
        self._texture = None
        self._config = None

    def get_current_texture(self):
        if not self._config:
            raise RuntimeError(
                "Canvas context must be configured before calling get_current_texture()."
            )

        if self._texture:
            return self._texture

        width, height = self._get_canvas().get_physical_size()
        width, height = max(width, 1), max(height, 1)

        self._texture = self._config["device"].create_texture(
            label="presentation-context",
            size=(width, height, 1),
            format=self._config["format"],
            usage=self._config["usage"],
        )
        return self._texture

    def present(self):
        if not self._texture:
            msg = "present() is called without a preceding call to "
            msg += "get_current_texture(). Note that present() is usually "
            msg += "called automatically after the draw function returns."
            raise RuntimeError(msg)
        else:
            texture = self._texture
            self._texture = None
            return self._get_canvas().present(texture)

    def get_preferred_format(self, adapter):
        canvas = self._get_canvas()
        if canvas:
            return canvas.get_preferred_format()
        else:
            return "rgba8unorm-srgb"


class WgpuOffscreenCanvasBase(WgpuCanvasBase):
    """Base class for off-screen canvases.

    It provides a custom context that renders to a texture instead of
    a surface/screen. On each draw the resulting image is passes as a
    texture to the ``present()`` method. Subclasses should (at least)
    implement ``present()``
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_surface_info(self):
        """This canvas does not correspond to an on-screen window."""
        return None

    def get_context(self, kind="webgpu"):
        """Get the GPUCanvasContext object to obtain a texture to render to."""
        # Normally this creates a GPUCanvasContext object provided by
        # the backend (e.g. wgpu-native), but here we use our own context.
        assert kind == "webgpu"
        if self._canvas_context is None:
            self._canvas_context = GPUCanvasContext(self)
        return self._canvas_context

    def present(self, texture):
        """Method that gets called at the end of each draw event.

        The rendered image is represented by the texture argument.
        Subclasses should overload this method and use the texture to
        process the rendered image.

        The texture is a new object at each draw, but is not explicitly
        destroyed, so it can be used e.g. as a texture binding (subject
        to set TextureUsage).
        """
        # Notes: Creating a new texture object for each draw is
        # consistent with how real canvas contexts work, plus it avoids
        # confusion of re-using the same texture except when the canvas
        # changes size. For use-cases where you do want to render to the
        # same texture one does not need the canvas API. E.g. in pygfx
        # the renderer can also work with a target that is a (fixed
        # size) texture.
        pass

    def get_preferred_format(self):
        """Get the preferred format for this canvas.

        This method can be overloaded to control the used texture
        format. The default is "rgba8unorm-srgb".
        """
        # Use rgba because that order is more common for processing and storage.
        # Use srgb because that's what how colors are usually expected to be.
        # Use 8unorm because 8bit is enough (when using srgb).
        return "rgba8unorm-srgb"


class WgpuManualOffscreenCanvas(WgpuAutoGui, WgpuOffscreenCanvasBase):
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

    def set_title(self, title):
        pass

    def close(self):
        self._closed = True

    def is_closed(self):
        return self._closed

    def _request_draw(self):
        # Deliberately a no-op, because people use .draw() instead.
        pass

    def present(self, texture):
        # This gets called at the end of a draw pass via GPUCanvasContext
        device = texture._device
        size = texture.size
        bytes_per_pixel = 4
        data = device.queue.read_texture(
            {
                "texture": texture,
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
        """Perform a draw and get the resulting image.

        The image array is returned as an NxMx4 memoryview object.
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
