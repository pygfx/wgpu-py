from flexx import flx
from pscript.stubs import window


raise NotImplementedError()


# Flexx cannot do multiple inheritance, so we consider WgpuCanvasBase an interface :)


class WgpuCanvas(flx.CanvasWidget):
    """Flexx widget to be used with the wgpu.backeds.js backend. Provides a
    canvas to render to.
    """

    def init(self):
        self._draw_func = None
        self._draw_pending = False

        self.node.addEventListener("size", self.request_draw)
        self.request_draw()

    def draw_frame(self):
        pass

    def _draw_frame_and_present(self):
        self._draw_pending = False
        self.draw_frame()

    def _request_draw(self):
        if not self._draw_pending:
            self._draw_pending = True
            window.requestAnimationFrame(self._draw_frame_and_present)

    def is_closed(self):
        return False
