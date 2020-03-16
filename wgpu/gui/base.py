import os
import sys
import logging
import ctypes.util

logger = logging.getLogger("wgpu")


class BaseCanvas:
    """ An abstract base canvas. Can be implementd to provide a canvas for
    various GUI toolkits.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._swap_chain = None
        self._err_hashes = {}
        self._display_id = None

    def configure_swap_chain(self, device, format, usage):
        """ Configures the swap chain for this canvas, and returns a
        new GPUSwapChain object representing it. Destroys any swapchain
        previously returned by configure_swap_chain, including all of the
        textures it has produced.
        """
        self._swap_chain = device._gui_configure_swap_chain(self, format, usage)
        return self._swap_chain

    def draw_frame(self):
        """ The function that gets called at each draw. You can implement
        this method in a subclass, or assign the attribute directly.
        """
        pass

    def _draw_frame_and_present(self):
        """ Draw the frame and present the swapchain. Errors are printed to stderr.
        Should be called by the subclass at an appropriate time.
        """
        # Perform the user-defined drawing code. When this errors,
        # we should report the error and then continue, otherwise we crash.
        try:
            self.draw_frame()
        except Exception as err:
            logger.error("Error during draw", exc_info=err)

        # Always present the swapchain, or wgpu gets into an error state.
        try:
            if self._swap_chain is not None:
                self._swap_chain._gui_present()  # a.k.a swap buffers
        except Exception as err:
            logger.error("Could not present the swapchain", exc_info=err)

    # Methods that must be overloaded

    def get_display_id(self):
        """ Get the native display id on Linux. This is needed by the backends
        to obtain a surface id on Linux.
        """
        # Re-use to avoid creating loads of id's
        if self._display_id is not None:
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

    def get_window_id(self):
        """ Get the native window id. This can be used by the backends
        to obtain a surface id.
        """
        raise NotImplementedError()

    def get_pixel_ratio(self):
        """ Get the float ratio between logical and physical pixels.
        """
        raise NotImplementedError()

    def get_logical_size(self):
        """ Get the logical size in float pixels.
        """
        raise NotImplementedError()

    def get_physical_size(self):
        """ Get the physical size in integer pixels.
        """
        raise NotImplementedError()

    def set_logical_size(self, width, height):
        """ Set the window size (in logical pixels).
        """
        raise NotImplementedError()

    def close(self):
        """ Close the window.
        """
        raise NotImplementedError()

    def is_closed(self):
        """ Whether the window is closed.
        """
        raise NotImplementedError()
