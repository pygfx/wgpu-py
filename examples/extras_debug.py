"""
Basic example of how to use wgpu-native instance extras to enable debug symbols.
As debugger we will use RenderDoc (https://renderdoc.org/) - other tools will require a similar setup.
While RenderDoc doesn't support WebGPU - it still works to inspect the Pipeline or with translated shaders.
"""

# test_example = false

from rendercanvas.auto import RenderCanvas, loop

# before we enumerate or request a device, we need to set the instance extras
# as we import from the base examples, those do the request_device call
from wgpu.backends.wgpu_native.extras import set_instance_extras

# this communicates with the compiler to enable debug symbols.
# I have confirmed this works for Vulkan and Dx12, however it seems like it's always enabled for Fxc and doesn't work for Dxc
# TODO can someone test this on Metal (OpenGL?)
set_instance_extras(
    flags=["Debug"]  # an additional option here is "Validation".
)

# TODO: replace the default examples by including additional debug markers using Encoder.inser_debug_marker(label)
try:
    from .cube import setup_drawing_sync
except ImportError:
    from cube import setup_drawing_sync


canvas = RenderCanvas(title="Cube example with debug symbols")
draw_frame = setup_drawing_sync(canvas)


@canvas.request_draw
def animate():
    draw_frame()
    canvas.request_draw()


if __name__ == "__main__":
    loop.run()
    # the first thing you might notice is that additional information is logged in your terminal.

# TODO maybe write this as a doc instead with screenshots perhaps?
# to launch this script using RenderDoc and capture a frame there use the following settings in the Launch Application tab:
# - Executable Path: python path
# - Working Directory: ~\wgpu-py\examples
# - Command Line Arguments: extras_debug.py
# - Environment Variables: - # can be left empty, `WGPU_BACKEND_TYPE=D3D12` for example works here
# make sure to check "Capture Child Processes"
# click on "Launch"! the GUI should start and you should see an overlay text in the frame telling you to hit F12 to capture a frame.
# Open the capture and on the left hand side in the Event Browser find the `vkCmdDrawIndexed` event. Click to open this event.
# Now the Pipeline State should let you chose either Vertex Shader or Fragment Shader and see a button called "View" next to the ShaderModule.
# If you see the source code from the example including symbols and comments then it worked!
# Note that using Dx12 will not show the the same source, as naga translated the shader to HLSL.

# TODO: automate the launching and capturing using the scripting api?
