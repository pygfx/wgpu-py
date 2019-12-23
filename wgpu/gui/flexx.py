from flexx import flx

from .base import BaseCanvas


# Flexx cannot do multiple inheritance, so we consider BaseCanvas an interface :)


class GpuWidget(flx.CanvasWidget):
    """ Flexx widget to be used with the wgpu.backed.js backend. Provider a
    canvas to render to.
    """

    def init(self):
        self._draw_func = None
        window.requestAnimationFrame(self._drawFrame)

    def configureSwapChain(self, *args):
        return self.node.configureSwapChain(*args)

    def setDrawFunction(self, func):
        self._draw_func = func

    def _drawFrame(self):
        window.requestAnimationFrame(self._drawFrame)
        if self._draw_func is not None:
            self._draw_func()
