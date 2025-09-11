from rendercanvas.glfw import RenderCanvas, loop
from .._coreutils import logger

logger.warning("The wgpu.gui.glfw is deprecated, use rendercanvas.glfw instead.")


WgpuCanvas = GlfwWgpuCanvas = RenderCanvas
run = loop.run
call_later = loop.call_later
