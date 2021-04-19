"""
Test the canvas, and parts of the rendering that involves a canvas,
like the swap chain.
"""

import os
import sys
import time

from pyshader import python2shader, vec4, i32
from pyshader import RES_INPUT, RES_OUTPUT
import wgpu.backends.rs  # noqa
from pytest import skip
from testutils import run_tests, can_use_wgpu_lib
from renderutils import render_to_texture, render_to_screen  # noqa


if os.getenv("CI") or not can_use_wgpu_lib:
    skip("Skipping tests that need a window or the wgpu lib", allow_module_level=True)


def setup_module():
    import glfw

    glfw.init()


def teardown_module():
    import glfw

    glfw.terminate()


def test_glfw_canvas_basics():
    """Create a window and check some of its behavior. No wgpu calls here."""

    import glfw
    from wgpu.gui.glfw import WgpuCanvas

    canvas = WgpuCanvas()

    canvas.set_logical_size(300, 200)
    glfw.poll_events()
    lsize = canvas.get_logical_size()
    assert isinstance(lsize, tuple) and len(lsize) == 2
    assert isinstance(lsize[0], float) and isinstance(lsize[1], float)
    assert lsize == (300.0, 200.0)

    assert len(canvas.get_physical_size()) == 2
    assert isinstance(canvas.get_pixel_ratio(), float)

    # Close
    assert not canvas.is_closed()
    if sys.platform.startswith("win"):  # On Linux we cant do this multiple times
        canvas.close()
        glfw.poll_events()
        assert canvas.is_closed()


@python2shader
def vertex_shader(
    index: (RES_INPUT, "VertexId", i32),
    pos: (RES_OUTPUT, "Position", vec4),
):
    positions = [
        vec3(-0.5, -0.5, 0.1),
        vec3(-0.5, +0.5, 0.1),
        vec3(+0.5, -0.5, 0.1),
        vec3(+0.5, +0.5, 0.1),
    ]
    p = positions[index]
    pos = vec4(p, 1.0)  # noqa


@python2shader
def fragment_shader(
    out_color: (RES_OUTPUT, 0, vec4),
):
    out_color = vec4(1.0, 0.5, 0.0, 1.0)  # noqa


def test_glfw_canvas_render():
    """Render an orange square ... in a glfw window."""

    import glfw
    from wgpu.gui.glfw import update_glfw_canvasses, WgpuCanvas

    canvas = WgpuCanvas()

    # wgpu.utils.get_default_device()
    adapter = wgpu.request_adapter(canvas=canvas, power_preference="high-performance")
    device = adapter.request_device()
    draw_frame1 = _get_draw_function(device, canvas)

    frame_counter = 0

    def draw_frame2():
        nonlocal frame_counter
        frame_counter += 1
        draw_frame1()

    canvas.request_draw(draw_frame2)

    # Give it a few rounds to start up
    for i in range(5):
        glfw.poll_events()
        update_glfw_canvasses()
    # There should have been exactly one draw now
    assert frame_counter == 1

    # Ask for a lot of draws
    for i in range(5):
        canvas.request_draw()
    # Process evens for a while
    for i in range(5):
        glfw.poll_events()
        update_glfw_canvasses()
    # We should have had just one draw
    assert frame_counter == 2

    # Change the canvase size
    canvas.set_logical_size(300, 200)
    canvas.set_logical_size(400, 300)
    for i in range(5):
        time.sleep(0.01)
        glfw.poll_events()
        update_glfw_canvasses()
    # We should have had just one draw
    assert frame_counter == 3

    # canvas.close()
    glfw.poll_events()


def test_glfw_canvas_render_custom_canvas():
    """Render an orange square ... in a glfw window. But not using WgpuCanvas.
    This helps make sure that WgpuCanvasInterface is indeed the minimal
    required canvas API.
    """

    import glfw

    class CustomCanvas:  # implements wgpu.WgpuCanvasInterface
        def __init__(self):
            glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
            glfw.window_hint(glfw.RESIZABLE, True)
            self.window = glfw.create_window(300, 200, "canvas", None, None)

        def get_window_id(self):
            if sys.platform.startswith("win"):
                return int(glfw.get_win32_window(self.window))
            elif sys.platform.startswith("darwin"):
                return int(glfw.get_cocoa_window(self.window))
            elif sys.platform.startswith("linux"):
                is_wayland = "wayland" in os.getenv("XDG_SESSION_TYPE", "").lower()
                if is_wayland:
                    return int(glfw.get_wayland_window(self.window))
                else:
                    return int(glfw.get_x11_window(self.window))
            else:
                raise RuntimeError(f"Cannot get GLFW window id on {sys.platform}.")

        def get_display_id(self):
            return wgpu.WgpuCanvasInterface.get_display_id(self)

        def get_physical_size(self):
            psize = glfw.get_framebuffer_size(self.window)
            return int(psize[0]), int(psize[1])

        def configure_swap_chain(self, *, device):
            format = "bgra8unorm-srgb"
            usage = wgpu.TextureUsage.RENDER_ATTACHMENT
            return wgpu.GPUCanvasContext.configure_swap_chain(
                self, device=device, format=format, usage=usage
            )

    canvas = CustomCanvas()

    adapter = wgpu.request_adapter(canvas=canvas, power_preference="high-performance")
    device = adapter.request_device()
    draw_frame = _get_draw_function(device, canvas)

    for i in range(5):
        time.sleep(0.01)
        glfw.poll_events()
        draw_frame()

    glfw.hide_window(canvas.window)


def _get_draw_function(device, canvas):
    # Bindings and layout
    bind_group_layout = device.create_bind_group_layout(entries=[])  # zero bindings
    bind_group = device.create_bind_group(layout=bind_group_layout, entries=[])
    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )

    vshader = device.create_shader_module(code=vertex_shader)
    fshader = device.create_shader_module(code=fragment_shader)

    render_pipeline = device.create_render_pipeline(
        label="my-debug-pipeline",
        layout=pipeline_layout,
        vertex={
            "module": vshader,
            "entry_point": "main",
            "buffers": [],
        },
        primitive={
            "topology": wgpu.PrimitiveTopology.triangle_strip,
            "strip_index_format": wgpu.IndexFormat.uint32,
            "front_face": wgpu.FrontFace.ccw,
            "cull_mode": wgpu.CullMode.none,
        },
        depth_stencil=None,
        multisample={
            "count": 1,
            "mask": 0xFFFFFFFF,
            "alpha_to_coverage_enabled": False,
        },
        fragment={
            "module": fshader,
            "entry_point": "main",
            "targets": [
                {
                    "format": wgpu.TextureFormat.bgra8unorm_srgb,
                    "blend": {
                        "color": (
                            wgpu.BlendFactor.one,
                            wgpu.BlendFactor.zero,
                            wgpu.BlendOperation.add,
                        ),
                        "alpha": (
                            wgpu.BlendFactor.one,
                            wgpu.BlendFactor.zero,
                            wgpu.BlendOperation.add,
                        ),
                    },
                },
            ],
        },
    )

    swap_chain = canvas.configure_swap_chain(device=device)

    def draw_frame():
        with swap_chain as current_texture_view:
            command_encoder = device.create_command_encoder()
            assert current_texture_view.size
            ca = {
                "view": current_texture_view,
                "resolve_target": None,
                "load_value": (0, 0, 0, 0),
                "store_op": wgpu.StoreOp.store,
            }
            render_pass = command_encoder.begin_render_pass(
                color_attachments=[ca],
            )

            render_pass.set_pipeline(render_pipeline)
            render_pass.set_bind_group(0, bind_group, [], 0, 999999)
            render_pass.draw(4, 1, 0, 0)
            render_pass.end_pass()
            device.queue.submit([command_encoder.finish()])

    return draw_frame


if __name__ == "__main__":
    setup_module()
    run_tests(globals())
    teardown_module()
