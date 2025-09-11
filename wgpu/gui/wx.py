from rendercanvas.wx import RenderCanvas, RenderWidget, loop
from .._coreutils import logger

logger.warning("The wgpu.gui.auto is deprecated, use rendercanvas.auto instead.")


WgpuWidget = WxWgpuWindow = RenderWidget
WgpuCanvas = WxWgpuCanvas = RenderCanvas
run = loop.run
call_later = loop.call_later
