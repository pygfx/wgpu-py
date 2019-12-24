from flexx import flx

from .base import BaseCanvas


# Flexx cannot do multiple inheritance, so we consider BaseCanvas an interface :)


class WgpuCanvas(flx.CanvasWidget):
    """ Flexx widget to be used with the wgpu.backend.js backend. Provides a
    canvas to render to.
    """

    def init(self):
        self._draw_func = None
        window.requestAnimationFrame(self._drawFrameAndPresent)

    def configureSwapChain(self, *args):
        return self.node.configureSwapChain(*args)

    def drawFrame(self):
        pass

    def _drawFrameAndPresent(self):
        window.requestAnimationFrame(self._drawFrameAndPresent)
        self.drawFrame()

    def isClosed(self):
        return False
