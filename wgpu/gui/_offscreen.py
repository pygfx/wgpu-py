from .. import base, flags
from ..gui.base import WgpuCanvasBase


class WgpuOffscreenCanvas(WgpuCanvasBase):
    """Base class for off-screen canvases.

    It provides a custom presentation context that renders to a tetxure
    instead of a surface/screen. The resulting texture view is passed
    to the `present()` method.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_window_id(self):
        """This canvas does not correspond to an on-screen window."""
        return None

    def get_context(self, kind="gpupresent"):
        """Get the GPUCanvasContext object to obtain a texture to render to."""
        # Normally this creates a GPUCanvasContext object provided by
        # the backend (e.g. rs), but here we use our own context.
        assert kind == "gpupresent"
        if self._canvas_context is None:
            self._canvas_context = GPUCanvasContextOffline(self)
        return self._canvas_context

    def present(self, texture_view):
        """Method that gets called at the end of each draw event. Subclasses
        should provide the approproate implementation.
        """
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


class GPUCanvasContextOffline(base.GPUCanvasContext):
    """Helper class for canvases that render to a texture."""

    def __init__(self, canvas):
        super().__init__(canvas)
        self._surface_size = (-1, -1)
        self._texture = None
        self._texture_view = None

    def unconfigure(self):
        super().unconfigure()
        # todo: maybe destroy texture? (currently API is unclear about destroying)
        self._texture = None

    def get_preferred_format(self, adapter):
        canvas = self._get_canvas()
        if canvas:
            return canvas.get_preferred_format()
        else:
            return "rgba8unorm-srgb"

    def get_current_texture(self):
        self._create_new_texture_if_needed()
        # todo: we return a view here, to align with the rs implementation, even though its wrong.
        return self._texture_view

    def present(self):
        if self._texture_view is not None:
            canvas = self._get_canvas()
            return canvas.present(self._texture_view)

    def _create_new_texture_if_needed(self):
        canvas = self._get_canvas()
        psize = canvas.get_physical_size()
        if psize == self._surface_size:
            return
        self._surface_size = psize

        self._texture = self._device.create_texture(
            label="presentation-context",
            size=(max(psize[0], 1), max(psize[1], 1), 1),
            format=self._format,
            usage=self._usage | flags.TextureUsage.COPY_SRC,
        )
        self._texture_view = self._texture.create_view()
