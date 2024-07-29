"""
Test occlusion queries.
"""

import numpy as np
import sys

import wgpu
from pytest import skip
from testutils import run_tests, get_default_device
from testutils import can_use_wgpu_lib, is_ci
from wgpu import flags

if not can_use_wgpu_lib:
    skip("Skipping tests that need the wgpu lib", allow_module_level=True)
elif is_ci and sys.platform == "win32":
    skip("These tests fail on dx12 for some reason", allow_module_level=True)


default_shader_source = """

struct Uniform {
    offset: vec3f,
    scale: vec2f,
}

@group(0) @binding(0) var<uniform> uniform : Uniform;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index : u32) -> @builtin(position) vec4<f32> {
    var positions = array<vec2f, 4>(
        vec2f(-0.5, -0.5),
        vec2f( 0.5, -0.5),
        vec2f(-0.5,  0.5),
        vec2f( 0.5,  0.5),
    );
    let p = positions[vertex_index];
    return vec4f(p * uniform.scale, 0.0, 1.0) + vec4f(uniform.offset, 0);
}
"""


def test_render_occluding_squares():
    device = get_default_device()

    # Bindings and layout
    bind_group_entries = [
        {"binding": 0, "visibility": flags.ShaderStage.VERTEX, "buffer": {}}
    ]
    bind_group_layout = device.create_bind_group_layout(entries=bind_group_entries)
    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )

    depth_texture = device.create_texture(
        size=[1024, 1024],
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
        format="depth32float",
    )

    shader = device.create_shader_module(code=default_shader_source)
    render_pipeline = device.create_render_pipeline(
        layout=pipeline_layout,
        vertex={
            "module": shader,
            "entry_point": "vs_main",
        },
        primitive={
            "topology": wgpu.PrimitiveTopology.triangle_strip,
            "cull_mode": wgpu.CullMode.back,
        },
        depth_stencil={
            "depth_write_enabled": True,
            "depth_compare": "less",
            "format": "depth32float",
        },
    )

    bind_groups = []
    expected_result = []

    # Each test draws a square to the screen centered at
    #    <x_offset, y_offset, z_offset + .5>
    # and where the sides of the square are length "side".  "Result" indicates
    # what we expect the result of the occlusion test to be
    def add_test(result, x_offset=0.0, y_offset=0.0, z=0.5, reverse=False):
        # See WGSL above for order.  Add padding.
        x_side = y_side = 0.1
        if reverse:
            y_side = -y_side
        data = np.float32((x_offset, y_offset, z, 0, x_side, y_side, 0, 0))
        buffer = device.create_buffer_with_data(
            data=data, usage=flags.BufferUsage.UNIFORM
        )
        binding = device.create_bind_group(
            layout=render_pipeline.get_bind_group_layout(0),
            entries=[{"binding": 0, "resource": {"buffer": buffer}}],
        )
        bind_groups.append(binding)
        expected_result.append(result)

    # These tests have to be run in the order shown, as some of the squares occlude
    # later squares.
    # small square in the center of the clipping area
    add_test(True)
    # small square in the corner of the clipping area, partially in, partially out
    add_test(True, x_offset=0.95, y_offset=0.95)
    # small square completely outside the clipping area.
    add_test(False, x_offset=2, y_offset=2)
    # Same as the first depth, but the previous big square completely occludes it
    add_test(False)
    # Same small square again, but bring it forward a little bit
    add_test(True, z=0.4)
    # Same small square, but so far forward that it's outside the clipping box
    add_test(False, z=-2)

    # Draw a square that should be visible, but it is culled because it is a rear-
    # facing rectangle. And to keep us honest, redraw the example again, but have it
    # face forward.
    add_test(False, x_offset=0.1, y_offset=0.1, reverse=True)
    add_test(True, x_offset=1, y_offset=0.1)

    occlusion_query_set = device.create_query_set(
        type="occlusion", count=len(bind_groups)
    )
    occlusion_buffer = device.create_buffer(
        size=len(bind_groups) * np.uint64().itemsize,
        usage=wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.QUERY_RESOLVE,
    )

    command_encoder = device.create_command_encoder()

    depth_stencil_attachment = {
        "view": depth_texture.create_view(),
        "depth_clear_value": 1.0,
        "depth_load_op": "clear",
        "depth_store_op": "store",
        "stencil_clear_value": 1.0,
        "stencil_load_op": "clear",
        "stencil_store_op": "store",
    }

    render_pass = command_encoder.begin_render_pass(
        color_attachments=[],
        depth_stencil_attachment=depth_stencil_attachment,
        occlusion_query_set=occlusion_query_set,
    )

    render_pass.set_pipeline(render_pipeline)
    # Draw each of the squares in the order given
    for index, binding in enumerate(bind_groups):
        render_pass.set_bind_group(0, binding)
        render_pass.begin_occlusion_query(index)
        render_pass.draw(4)
        render_pass.end_occlusion_query()
    render_pass.end()
    # Get the result of the occlusion test
    command_encoder.resolve_query_set(
        occlusion_query_set, 0, len(bind_groups), occlusion_buffer, 0
    )
    device.queue.submit([command_encoder.finish()])

    memory_view = device.queue.read_buffer(occlusion_buffer)
    array = np.frombuffer(memory_view, dtype=np.uint64)
    # https://www.w3.org/TR/webgpu/#occlusion
    # Any non-zero value indicates that at least one sample passed.
    actual_result = [x != 0 for x in array]
    assert actual_result == expected_result


if __name__ == "__main__":
    run_tests(globals())
