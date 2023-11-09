"""
Test creation of offscreen canvas windows.
"""

import wgpu
import pytest
import testutils
from testutils import can_use_wgpu_lib, create_and_release


if not can_use_wgpu_lib:
    pytest.skip("Skipping tests that need wgpu lib", allow_module_level=True)


DEVICE = wgpu.utils.get_default_device()


def make_draw_func_for_canvas(canvas):
    """Create a draw function for the given canvas,
    so that we can really present something to a canvas being tested.
    """
    ctx = canvas.get_context()
    ctx.configure(device=DEVICE, format="bgra8unorm-srgb")

    def draw():
        ctx = canvas.get_context()
        command_encoder = DEVICE.create_command_encoder()
        current_texture_view = ctx.get_current_texture()
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": current_texture_view,
                    "resolve_target": None,
                    "clear_value": (1, 1, 1, 1),
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                }
            ],
        )
        render_pass.end()
        DEVICE.queue.submit([command_encoder.finish()])
        ctx.present()

    return draw


@create_and_release
def test_release_canvas_context(n):
    # Test with offscreen canvases. A context is created, but not a wgpu-native surface.

    # Note: the offscreen canvas keeps the render-texture-view alive, since it
    # is used to e.g. download the resulting image. That's why we also see
    # Textures and TextureViews in the counts.

    from wgpu.gui.offscreen import WgpuCanvas

    yield {
        "expected_counts_after_create": {
            "CanvasContext": (n, 0),
            "Texture": (n, n),
            "TextureView": (n, n),
        },
    }

    for i in range(n):
        c = WgpuCanvas()
        c.request_draw(make_draw_func_for_canvas(c))
        c.draw()
        yield c.get_context()


TEST_FUNCS = [test_release_canvas_context]


if __name__ == "__main__":
    # Set to true and run as script to do a memory stress test
    testutils.TEST_MEM_USAGE = False

    test_release_canvas_context()
