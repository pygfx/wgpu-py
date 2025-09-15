from rendercanvas.offscreen import RenderCanvas, loop
from .._coreutils import logger

logger.warning(
    "The wgpu.gui.offscreen is deprecated, use rendercanvas.offscreen instead."
)


WgpuCanvas = WgpuManualOffscreenCanvas = RenderCanvas
run = loop.run
call_later = loop.call_later
