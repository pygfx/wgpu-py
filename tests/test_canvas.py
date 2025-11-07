"""Test that wgpu works together with rendercanvas."""

import wgpu

# from rendercanvas import BaseRenderCanvas
from rendercanvas.offscreen import RenderCanvas

from pytest import skip
from testutils import run_tests, can_use_wgpu_lib


if not can_use_wgpu_lib:
    skip("Skipping tests that need the wgpu lib", allow_module_level=True)


shader_source = """
@vertex
fn vs_main(@builtin(vertex_index) vertex_index : u32) -> @builtin(position) vec4<f32> {
    var positions: array<vec2<f32>, 3> = array<vec2<f32>, 3>(vec2<f32>(0.0, -0.5), vec2<f32>(0.5, 0.5), vec2<f32>(-0.5, 0.7));
    let p: vec2<f32> = positions[vertex_index];
    return vec4<f32>(p, 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.5, 0.0, 1.0);
}
"""


def test_rendercanvas():
    """Render an orange square ... in an offscreen RenderCanvas

    If this works, we can assume the other rendercanvas backends work too.
    """

    canvas = RenderCanvas(size=(640, 480))

    device = wgpu.utils.get_default_device()
    draw_frame1 = _get_draw_function(device, canvas.get_wgpu_context())

    frame_counter = 0

    def draw_frame2():
        nonlocal frame_counter
        frame_counter += 1
        draw_frame1()

    canvas.request_draw(draw_frame2)

    m = canvas.draw()
    assert isinstance(m, memoryview)
    assert m.shape == (480, 640, 4)
    assert frame_counter == 1

    for i in range(5):
        canvas.draw()

    assert frame_counter == 6

    # Change the canvas size
    canvas.set_logical_size(300, 200)
    m = canvas.draw()
    assert m.shape == (200, 300, 4)


def _get_draw_function(device, present_context):
    # Bindings and layout
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[])

    shader = device.create_shader_module(code=shader_source)

    render_texture_format = present_context.get_preferred_format(device.adapter)
    present_context.configure(device=device, format=render_texture_format)

    render_pipeline = device.create_render_pipeline(
        label="my-debug-pipeline",
        layout=pipeline_layout,
        vertex={
            "module": shader,
            "entry_point": "vs_main",
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
            "module": shader,
            "entry_point": "fs_main",
            "targets": [
                {
                    "format": render_texture_format,
                    "blend": {
                        "color": {},  # use defaults
                        "alpha": {},  # use defaults
                    },
                },
            ],
        },
    )

    def draw_frame():
        current_texture_view = present_context.get_current_texture().create_view()
        command_encoder = device.create_command_encoder()
        assert current_texture_view.size
        ca = {
            "view": current_texture_view,
            "resolve_target": None,
            "clear_value": (0, 0, 0, 0),
            "load_op": wgpu.LoadOp.clear,
            "store_op": wgpu.StoreOp.store,
        }
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[ca],
        )

        render_pass.set_pipeline(render_pipeline)
        render_pass.draw(4, 1, 0, 0)
        render_pass.end()
        device.queue.submit([command_encoder.finish()])

    return draw_frame


if __name__ == "__main__":
    run_tests(globals())
