from rendercanvas.qt import RenderCanvas, RenderWidget, loop
from .._coreutils import logger

logger.warning(
    "The wgpu.gui.qt is deprecated, use rendercanvas.qt instead (or .pyside or .pyqt6)."
)


WgpuWidget = QWgpuWidget = RenderWidget
WgpuCanvas = QWgpuCanvas = RenderCanvas
run = loop.run
call_later = loop.call_later
