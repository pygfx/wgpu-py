from flexx import flx
from pscript.stubs import window


# Flexx cannot do multiple inheritance, so we consider BaseCanvas an interface :)


class WgpuCanvas(flx.CanvasWidget):
    """ Flexx widget to be used with the wgpu.backed.js backend. Provides a
    canvas to render to.
    """

    def init(self):
        self._draw_func = None
        window.requestAnimationFrame(self._draw_frame_and_present)

    def configure_swap_chain(self, *args):
        return self.node.configureSwapChain(*args)

    def draw_frame(self):
        pass

    def _draw_frame_and_present(self):
        window.requestAnimationFrame(self._draw_frame_and_present)
        self.draw_frame()

    def is_closed(self):
        return False
