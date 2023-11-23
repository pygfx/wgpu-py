from .. import classes, flags
from ..gui.base import WgpuCanvasBase


class WgpuOffscreenCanvas(WgpuCanvasBase):
    """Base class for off-screen canvases.

    It provides a custom context that renders to a texture instead of
    a surface/screen. The resulting texture is passed to the `present()`
    method.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_window_id(self):
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
        process the rendered image. The texture is a new object at each
        draw, but is not explicitly destroyed, so it can be used e.g.
        as a texture binding (subject to set TextureUsage).
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
        """Get the preferred format for this canvas. This method can
        be overloaded to control the used texture format. The default
        is "rgba8unorm-srgb".
        """
        # Use rgba because that order is more common for processing and storage.
        # Use srgb because that's what how colors are usually expected to be.
        # Use 8unorm because 8bit is enough (when using srgb).
        return "rgba8unorm-srgb"


class GPUCanvasContext(classes.GPUCanvasContext):
    """Helper class for canvases that render to a texture."""

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
            msg = "present() is called without a preceeding call to "
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
