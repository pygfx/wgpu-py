import os
import sys
import logging
import ctypes.util

from .. import base, GPUDevice, flags, enums


logger = logging.getLogger("wgpu")


class WgpuCanvasInterface(base.GPUCanvasContext):
    """This is the interface that a canvas object must implement in order
    to be a valid canvas that wgpu can work with.
    """

    # NOTE: It is not necessary to actually subclass this class.

    # Whether to present to a surface or to a texture
    _PRESENT_TO_SURFACE = True

    def __init__(self, *args, **kwargs):
        # The args/kwargs are there because we may be mixed with e.g. a Qt widget
        super().__init__(*args, **kwargs)

    def get_window_id(self):
        """Get the native window id. This is used by the backends
        to obtain a surface id.
        """
        raise NotImplementedError()

    def get_display_id(self):
        """Get the native display id on Linux. This is used by the backends
        to obtain a surface id on Linux. The default implementation calls into
        the X11 lib to get the display id.
        """
        # Re-use to avoid creating loads of id's
        if getattr(self, "_display_id", None) is not None:
            return self._display_id

        if sys.platform.startswith("linux"):
            is_wayland = "wayland" in os.getenv("XDG_SESSION_TYPE", "").lower()
            if is_wayland:
                raise NotImplementedError(
                    f"Cannot (yet) get display id on {self.__class__.__name__}."
                )
            else:
                x11 = ctypes.CDLL(ctypes.util.find_library("X11"))
                x11.XOpenDisplay.restype = ctypes.c_void_p
                self._display_id = x11.XOpenDisplay(None)
        else:
            raise RuntimeError(f"Cannot get display id on {sys.platform}.")

        return self._display_id

    def get_physical_size(self):
        """Get the physical size in integer pixels."""
        raise NotImplementedError()

    def configure_swap_chain(
        self,
        *,
        label="",
        device: "GPUDevice",
        format: "enums.TextureFormat" = None,
        usage: "flags.TextureUsage" = None,
    ):
        """Obtain a swap-chain object."""
        # Let's be nice and allow not-specifying the format
        format = format or self.get_swap_chain_preferred_format(device.adapter)
        return super().configure_swap_chain(
            label=label, device=device, format=format, usage=usage
        )

    def get_swap_chain_preferred_format(self, adapter):
        """Get the preferred swap-chain texture format for this canvas."""
        return "bgra8unorm-srgb"  # seems to be a good default, can be overridden

    def _present(self, texture_view):
        """Offscreen canvases must implement this and set _PRESENT_TO_SURFACE to False."""
        raise NotImplementedError()


class WgpuCanvasBase(WgpuCanvasInterface):
    """An abstract class, extending :class:`WgpuCanvasInterface`,
    that provides a base canvas for various GUI toolkits, so
    that basic canvas functionality is available via a common API.

    It convenient, but not required to use this class (or any of its
    subclasses) to use wgpu-py.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._err_hashes = {}

    def draw_frame(self):
        """The function that gets called at each draw. You can implement
        this method in a subclass, or set it via a call to request_draw().
        """
        pass

    def request_draw(self, draw_function=None):
        """Request from the main loop to schedule a new draw event,
        so that the canvas will be updated. If draw_function is not
        given, the last set drawing function is used.
        """
        if draw_function is not None:
            self.draw_frame = draw_function
        self._request_draw()

    def _draw_frame_and_present(self):
        """Draw the frame and present the swapchain. Errors are logged to the
        "wgpu" logger. Should be called by the subclass at an appropriate time.
        """
        # Perform the user-defined drawing code. When this errors,
        # we should report the error and then continue, otherwise we crash.
        try:
            self.draw_frame()
        except Exception as err:
            self._log_exception("Draw error", err)

    def _log_exception(self, kind, err):
        """Log the given exception instance, but only log a one-liner for
        subsequent occurances of the same error to avoid spamming (which
        can happen easily with errors in the drawing code).
        """
        msg = str(err)
        msgh = hash(msg)
        if msgh not in self._err_hashes:
            # Provide the exception, so the default logger prints a stacktrace.
            # IDE's can get the exception from the root logger for PM debugging.
            self._err_hashes[msgh] = 1
            logger.error(kind, exc_info=err)
        else:
            # We've seen this message before, return a one-liner instead.
            self._err_hashes[msgh] = count = self._err_hashes[msgh] + 1
            msg = kind + ": " + msg.split("\n")[0].strip()
            msg = msg if len(msg) <= 70 else msg[:69] + "…"
            logger.error(msg + f" ({count})")

    # Methods that must be overloaded

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

    def close(self):
        """Close the window."""
        raise NotImplementedError()

    def is_closed(self):
        """Get whether the window is closed."""
        raise NotImplementedError()

    def _request_draw(self):
        """This should invoke a new draw in a later event loop
        iteration (i.e. the call itself should return directly).
        Multiple calls should result in a single new draw. Preferably
        the FPS is limited to avoid draining CPU and power.
        """
        raise NotImplementedError()


class GPUSwapChainOffScreen(base.GPUSwapChain):
    """Helper class for canvases that render to a texture."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._surface_size = (-1, -1)
        self._texture = None

    def _create_new_texture_if_needed(self):
        canvas = self._canvas
        psize = canvas.get_physical_size()
        if psize == self._surface_size:
            return
        self._surface_size = psize

        self._texture = self._device.create_texture(
            label="swapchain",
            size=(max(psize[0], 1), max(psize[1], 1), 1),
            format=self._format,
            usage=self._usage | flags.TextureUsage.COPY_SRC,
        )
        self._texture_view = self._texture.create_view()

    def __enter__(self):
        # Get the current texture view, and make sure it is presented when done
        self._create_new_texture_if_needed()
        return self._texture_view

    def __exit__(self, type, value, tb):
        self._canvas._present(self._texture_view)


# todo: ugly hack to avoid circular import. Fix that when we refactor swap chain
base.GPUSwapChainOffScreen = GPUSwapChainOffScreen
