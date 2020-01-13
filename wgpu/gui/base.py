import os
import sys
import traceback
import ctypes.util


class BaseCanvas:
    """ An abstract base canvas. Can be implementd to provide a canvas for
    various GUI toolkits.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._swapchain = None
        self._err_hashes = {}
        self._display_id = None

    def configureSwapChain(self, device, format, usage):
        """ Configures the swap chain for this canvas, and returns a
        new GPUSwapChain object representing it. Destroys any swapchain
        previously returned by configureSwapChain, including all of the
        textures it has produced.
        """
        self._swapchain = device._gui_configureSwapChain(self, format, usage)
        return self._swapchain

    def drawFrame(self):
        """ The function that gets called at each draw. You can implement
        this method in a subclass, or assign the attribute directly.
        """
        pass

    def _drawFrameAndPresent(self):
        """ Draw the frame and present the swapchain. Errors are printed to stderr.
        Should be called by the subclass at an appropriate time.
        """
        try:
            self.drawFrame()
            if self._swapchain is not None:
                self._swapchain._gui_present()  # a.k.a swap buffers
        except Exception:
            # Enable PM debuging
            sys.last_type, sys.last_value, sys.last_traceback = sys.exc_info()
            msg = str(sys.last_value)
            msgh = hash(msg)
            if msgh in self._err_hashes:
                count = self._err_hashes[msgh] + 1
                self._err_hashes[msgh] = count
                shortmsg = msg.split("\n", 1)[0].strip()[:50]
                sys.stderr.write(f"Error in draw again ({count}): {shortmsg}\n")
            else:
                self._err_hashes[msgh] = 1
                sys.stderr.write(f"Error in draw: " + msg.strip() + "\n")
                traceback.print_last(6)

    # Methods that must be overloaded

    def getSizeAndPixelRatio(self):
        """ Get a three-element tuple (width, height, pixelratio). This
        can be used internally (by the backends) to create the
        swapchain, and by users to determine the canvas size.
        """
        raise NotImplementedError()

    def isClosed(self):
        """ Whether the window is closed.
        """
        raise NotImplementedError()

    def getWindowId(self):
        """ Get the native window id. This can be used by the backends
        to obtain a surface id.
        """
        raise NotImplementedError()

    def getDisplayId(self):
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
