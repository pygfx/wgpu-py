import sys
import traceback


class BaseCanvas:
    """ An abstract base canvas. Can be implementd to provide a canvas for
    various GUI toolkits.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._swapchain = None
        self._err_hashes = {}

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
