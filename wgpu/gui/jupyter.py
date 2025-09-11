from rendercanvas.jupyter import RenderCanvas, loop
from .._coreutils import logger

logger.warning("The wgpu.gui.jupyter is deprecated, use rendercanvas.jupyter instead.")


WgpuCanvas = JupyterWgpuCanvas = RenderCanvas
run = loop.run
call_later = loop.call_later
