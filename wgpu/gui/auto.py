from rendercanvas.auto import RenderCanvas, loop
from .._coreutils import logger

logger.warning("The wgpu.gui.auto is deprecated, use rendercanvas.auto instead.")


WgpuCanvas = RenderCanvas
run = loop.run
call_later = loop.call_later
