import wgpu


class WgpuCanvasInterface:
    """The minimal interface to be a valid canvas that wgpu can render to.

    Any object that implements these methods is a canvas that wgpu can work with.
    The object does not even have to derive from this class.
    In practice, we recommend using the `rendercanvas <https://github.com/pygfx/rendercanvas>`_ library.
    """

    # This implementation serves as documentation, but it actually works!

    _canvas_context = None

    def get_physical_size(self) -> tuple[int, int]:
        """Get the physical size of the canvas in integer pixels."""
        return (640, 480)

    def get_context(self, context_type: str = "wgpu") -> wgpu.GPUCanvasContext:
        """Get the ``GPUCanvasContext`` object corresponding to this canvas.

        The context is used to obtain a texture to render to, and to
        present that texture to the canvas.

        The canvas should get the context once, and then store it on ``self``.
        Getting the context is best done using ``wgpu.rendercanvas_context_hook()``,
        which accepts two arguments: the canvas object, and a dict with the present-methods
        that this canvas supports.

        Each supported present-method is represented by a field in the dict. The value
        is another dict with information specific to that present method.
        A canvas must implement at least either the "screen" or "bitmap" method.

        With method "screen", the context will render directly to a surface
        representing the region on the screen. The sub-dict should have a ``window``
        field containing the window id. On Linux there should also be ``platform``
        field to distinguish between "wayland" and "x11", and a ``display`` field
        for the display id. This information is used by wgpu to obtain the required
        surface id.

        With method "bitmap", the context will present the result as an image
        bitmap. On GPU-based contexts, the result will first be rendered to an
        offscreen texture, and then downloaded to RAM. The sub-dict must have a
        field 'formats': a list of supported image formats. Examples are "rgba-u8"
        and "i-u8". A canvas must support at least "rgba-u8". Note that srgb mapping
        is assumed to be handled by the canvas.

        Also see https://rendercanvas.readthedocs.io/stable/contextapi.html
        """

        # Note that this function is analog to HtmlCanvas.getContext(), except
        # here the only valid arg is 'webgpu', which is also made the default.
        assert context_type in ("wgpu", "webgpu", None)

        # Support only bitmap-present, with rgba8unorm.
        present_methods = {
            "bitmap": {
                "formats": ["rgba-u8"],
            }
        }

        if self._canvas_context is None:
            self._canvas_context = wgpu.rendercanvas_context_hook(self, present_methods)

        return self._canvas_context
