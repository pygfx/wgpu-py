"""
Test statistics queries.
"""

import sys

import numpy as np
import pytest
from pytest import skip

import wgpu
from testutils import can_use_wgpu_lib, is_ci, run_tests
from wgpu import TextureFormat
from wgpu.backends.wgpu_native.extras import (
    PipelineStatisticName,
    begin_pipeline_statistics_query,
    create_statistics_query_set,
    end_pipeline_statistics_query,
)

if not can_use_wgpu_lib:
    skip("Skipping tests that need the wgpu lib", allow_module_level=True)
elif is_ci and sys.platform == "win32":
    skip("These tests fail on dx12 for some reason", allow_module_level=True)


default_shader_source = """

// Draws a square with side 0.1 centered at the indicated location.
// If reverse, we take the vertices clockwise rather than counterclockwise so that
// we can test culling.

@vertex
fn vertex(@builtin(vertex_index) vertex_index : u32,
           @builtin(instance_index) instance_index : u32
) -> @builtin(position) vec4<f32> {
    var positions = array<vec2f, 4>(
        vec2f(-0.05, -0.05),
        vec2f( 0.05, -0.05),
        vec2f(-0.05,  0.05),
        vec2f( 0.05,  0.05),
    );
    var p = positions[vertex_index];
    if instance_index == 1 {
        // Swapping x and y will cause the coordinates to be cw instead of ccw
        p = vec2f(p.y, p.x);
    }
    return vec4f(p, 0.0, 1.0);
}

@fragment
fn fragment( ) -> @location(0) vec4f {
    return vec4f();
}

@compute @workgroup_size(64)
fn compute() {
}
"""


def test_render_occluding_squares():
    adapter = wgpu.gpu.request_adapter(power_preference="high-performance")
    try:
        device = adapter.request_device(required_features=["pipeline-statistics-query"])
    except RuntimeError:
        pytest.skip("pipeline-statistics-query not supported")

    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[])

    output_texture = device.create_texture(
        size=[1024, 1024],
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
        format=TextureFormat.rgba8unorm,
    )

    shader = device.create_shader_module(code=default_shader_source)
    render_pipeline = device.create_render_pipeline(
        layout=pipeline_layout,
        vertex={
            "module": shader,
        },
        fragment={
            "module": shader,
            "targets": [{"format": output_texture.format}],
        },
        primitive={
            "topology": wgpu.PrimitiveTopology.triangle_strip,
            "cull_mode": wgpu.CullMode.back,
        },
    )

    compute_pipeline = device.create_compute_pipeline(
        layout=pipeline_layout,
        compute={"module": shader},
    )

    color_attachment = {
        "clear_value": (0, 0, 0, 0),  # only first value matters
        "load_op": "clear",
        "store_op": "store",
        "view": output_texture.create_view(),
    }

    occlusion_query_set = create_statistics_query_set(
        device,
        count=2,
        statistics=[
            "vertex-shader-invocations",  # name can be snake case string
            "ClipperInvocations",  # name can be CamelCase
            "clipper-primitives-out",
            "fragment_shader_invocations",  # name can have underscores
            PipelineStatisticName.ComputeShaderInvocations,  # and there's an enum.
        ],
    )
    occlusion_buffer = device.create_buffer(
        size=2 * 5 * np.uint64().itemsize,
        usage=wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.QUERY_RESOLVE,
    )

    command_encoder = device.create_command_encoder()

    render_pass = command_encoder.begin_render_pass(
        color_attachments=[color_attachment]
    )
    begin_pipeline_statistics_query(render_pass, occlusion_query_set, 0)
    render_pass.set_pipeline(render_pipeline)
    render_pass.draw(4, 2)
    end_pipeline_statistics_query(render_pass)
    render_pass.end()

    compute_pass = command_encoder.begin_compute_pass()
    begin_pipeline_statistics_query(compute_pass, occlusion_query_set, 1)
    compute_pass.set_pipeline(compute_pipeline)
    compute_pass.dispatch_workgroups(10)
    end_pipeline_statistics_query(compute_pass)
    compute_pass.end()

    command_encoder.resolve_query_set(occlusion_query_set, 0, 2, occlusion_buffer, 0)
    device.queue.submit([command_encoder.finish()])

    render_result = (
        device.queue.read_buffer(occlusion_buffer, size=40).cast("Q").tolist()
    )
    compute_result = (
        device.queue.read_buffer(occlusion_buffer, buffer_offset=40).cast("Q").tolist()
    )

    # We know that compute was called 10 * 60 times, exactly
    assert compute_result == [0, 0, 0, 0, 10 * 64]
    assert render_result[0] == 8  # 4 vertices, 2 instances
    assert render_result[1] == 4  # 4 triangles
    # unclear what exactly render_result[2] is.
    assert render_result[3] > 1000
    assert render_result[4] == 0  # no calls to the compute engine


def test_enum_is_in_sync():
    """
    The enum PipelineStatisticsName is created by hand, while the enum_str2int value
    is generated automatically from wgpu.h.  They should both contain the same strings.
    If this test fails, their values have diverged.

    Either fix PipelineStatisticsName or modify this test and explain what the difference
    is.
    """
    from wgpu.backends.wgpu_native._mappings import enum_str2int

    enum_list = set(PipelineStatisticName)
    native_list = set(enum_str2int["PipelineStatisticName"].keys())
    assert enum_list == native_list


if __name__ == "__main__":
    run_tests(globals())
