"""
Test the canvas, and parts of the rendering that involves a canvas,
like the swap chain.
"""

import os

from python_shader import python2shader, vec4, i32
from python_shader import RES_INPUT, RES_OUTPUT
import wgpu.backends.rs  # noqa
from pytest import skip
from testutils import can_use_wgpu_lib, get_default_device
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
    """ Create a window and check some of its behavior. No wgpu calls here.
    """

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
    canvas.close()
    glfw.poll_events()
    assert canvas.is_closed()


@python2shader
def vertex_shader(
    index: (RES_INPUT, "VertexId", i32), pos: (RES_OUTPUT, "Position", vec4),
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
def fragment_shader(out_color: (RES_OUTPUT, 0, vec4),):
    out_color = vec4(1.0, 0.5, 0.0, 1.0)  # noqa


def test_glfw_canvas_render():
    """ Render an orange square ... in a glfw window.
    """

    import glfw
    from wgpu.gui.glfw import update_glfw_canvasses, WgpuCanvas

    canvas = WgpuCanvas()
    device = get_default_device()

    # Bindings and layout
    bind_group_layout = device.create_bind_group_layout(entries=[])  # zero bindings
    bind_group = device.create_bind_group(layout=bind_group_layout, entries=[])
    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )

    vshader = device.create_shader_module(code=vertex_shader)
    fshader = device.create_shader_module(code=fragment_shader)

    render_pipeline = device.create_render_pipeline(
        layout=pipeline_layout,
        vertex_stage={"module": vshader, "entry_point": "main"},
        fragment_stage={"module": fshader, "entry_point": "main"},
        primitive_topology=wgpu.PrimitiveTopology.triangle_strip,
        rasterization_state={
            "front_face": wgpu.FrontFace.ccw,
            "cull_mode": wgpu.CullMode.none,
            "depth_bias": 0,
            "depth_bias_slope_scale": 0.0,
            "depth_bias_clamp": 0.0,
        },
        color_states=[
            {
                "format": wgpu.TextureFormat.bgra8unorm_srgb,
                "alpha_blend": (
                    wgpu.BlendFactor.one,
                    wgpu.BlendFactor.zero,
                    wgpu.BlendOperation.add,
                ),
                "color_blend": (
                    wgpu.BlendFactor.one,
                    wgpu.BlendFactor.zero,
                    wgpu.BlendOperation.add,
                ),
                "write_mask": wgpu.ColorWrite.ALL,
            }
        ],
        depth_stencil_state=None,
        vertex_state={"index_format": wgpu.IndexFormat.uint32, "vertex_buffers": [],},
        sample_count=1,
        sample_mask=0xFFFFFFFF,
        alpha_to_coverage_enabled=False,
    )

    swap_chain = canvas.configure_swap_chain(
        device,
        canvas.get_swap_chain_preferred_format(device),
        wgpu.TextureUsage.OUTPUT_ATTACHMENT,
    )

    frame_counter = 0

    def draw_frame():
        nonlocal frame_counter
        frame_counter += 1

        current_texture_view = swap_chain.get_current_texture_view()
        command_encoder = device.create_command_encoder()

        ca = {
            "attachment": current_texture_view,
            "resolve_target": None,
            "load_value": (0, 0, 0, 0),
            "store_op": wgpu.StoreOp.store,
        }
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[ca], depth_stencil_attachment=None,
        )

        render_pass.set_pipeline(render_pipeline)
        render_pass.set_bind_group(0, bind_group, [], 0, 999999)
        render_pass.draw(4, 1, 0, 0)
        render_pass.end_pass()
        device.default_queue.submit([command_encoder.finish()])

    canvas.draw_frame = draw_frame

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
        glfw.poll_events()
        update_glfw_canvasses()
    # We should have had just one draw
    assert frame_counter == 3

    canvas.close()
    glfw.poll_events()


if __name__ == "__main__":
    setup_module()

    test_glfw_canvas_basics()
    test_glfw_canvas_render()

    teardown_module()
