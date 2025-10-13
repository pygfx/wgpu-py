"""
Simple example to show how the wgpu-native extras can be used to use dxc compiler for DX12.
Since this will only work on Windows it's not meant for the test suite.
You can run the download script using `python tools/download_dxc.py` to download the latest Dxc release from GitHub. And extract it to the resource directory.
"""

# run_example = false

from rendercanvas.auto import RenderCanvas, loop

# before we enumerate or request a device, we need to set the instance extras
# as we import from the base examples, those do the request_device call
from wgpu.backends.wgpu_native.extras import set_instance_extras

set_instance_extras(
    backends=[
        "DX12"
    ],  # using the env var `WGPU_BACKEND_TYPE` happens later during request_device, so you can only select backends that are requested for the instance
    dx12_compiler="Dxc",  # request the Dxc compiler to be used
    # dxc_path can be set for a custom Dxc location
    dxc_max_shader_model=6.7,
    # by setting these limits to percentages 0..100 you will get a Validation Error, should too much memory be requested.
    budget_for_device_creation=99,
    budget_for_device_loss=97,
)


try:
    from .cube import setup_drawing_sync
except ImportError:
    from cube import setup_drawing_sync


canvas = RenderCanvas(title="Cube example on DX12 using Dxc")
draw_frame = setup_drawing_sync(canvas)


@canvas.request_draw
def animate():
    draw_frame()
    canvas.request_draw()


if __name__ == "__main__":
    loop.run()
    # But how do you know this is actually using Dxc over Fxc?
    # perhaps performance, but we can also use Debug tools to be sure.
