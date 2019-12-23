class BaseCanvas:
    """ An abstract base canvas. Can be implementd to provide a canvas for
    various GUI toolkits.
    """

    _swapchain = None
    _drawFunction = None

    def getSizeAndPixelRatio(self):
        """ Get a three-element tuple (width, height, pixelratio). This
        can be used internally (by the backends) to create the
        swapchain, and by users to determine the canvas size.
        """
        raise NotImplementedError()

    def getWindowId(self):
        """ Get the native window id. This can be used by the backends
        to obtain a surface id.
        """
        raise NotImplementedError()

    def configureSwapChain(self, device, format, usage):
        """ Configures the swap chain for this canvas, and returns a
        new GPUSwapChain object representing it. Destroys any swapchain
        previously returned by configureSwapChain, including all of the
        textures it has produced.
        """
        self._swapchain = device._gui_configureSwapChain(self, format, usage)  # noqa
        return self._swapchain

    def setDrawFunction(self, func):
        """ Set the function to call at each draw. This function will be called
        automatically whenever the canvas needs to be redrawn.
        """
        self._drawFunction = func
