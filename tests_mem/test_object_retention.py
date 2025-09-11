import struct

import pytest

import wgpu
from wgpu.backends.wgpu_native.extras import (
    PipelineStatisticName,
    begin_pipeline_statistics_query,
    create_statistics_query_set,
    end_pipeline_statistics_query,
)

#
# wgpu is generally good at keeping alive those objects that it needs to complete a
# GPU operation.  This file has tests that agressively deletes objects (so that their
# release method will be called) even though the GPU has pending operations on them.
# We confirm that the operations complete successfully.

SHADER_SOURCE = """
    @group(0) @binding(0) var<uniform> offset: vec2f;

    @vertex
    fn vertex_main( @location(0) position: vec2f ) -> @builtin(position) vec4f {
        return vec4f(position + offset, 0, 1.0);
    }

    @fragment
    fn fragment_main() -> @location(0) vec4f {
        return vec4f();
    }

    @compute
    @workgroup_size(1)
    fn main(@builtin(global_invocation_id) index: vec3<u32>) {
        _ = offset;
    }
"""

VERTEX_BUFFER_LAYOUT = {
    "array_stride": 4 * 2,
    "attributes": [
        {
            "format": wgpu.VertexFormat.float32x2,
            "offset": 0,
            "shader_location": 0,
        },
    ],
}

BIND_GROUP_ENTRIES = [
    {
        "binding": 0,
        "visibility": wgpu.flags.ShaderStage.VERTEX | wgpu.flags.ShaderStage.COMPUTE,
        "buffer": {},
    }
]


@pytest.mark.parametrize("use_render_bundle", [False, True])
def test_object_retention_in_render(use_render_bundle, capsys):
    with capsys.disabled():
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        desired_features = ["timestamp-query", "pipeline-statistics-query"]
        features = [x for x in desired_features if x in adapter.features]
        device = adapter.request_device_sync(required_features=features)
        has_timestamps = "timestamp-query" in features
        has_statistics = "pipeline-statistics-query" in features

        output_texture_format = wgpu.TextureFormat.rgba8unorm
        depth_texture_format = wgpu.TextureFormat.depth32float

        shader = device.create_shader_module(code=SHADER_SOURCE)

        bind_group_entries = BIND_GROUP_ENTRIES.copy()
        bind_group_layout = device.create_bind_group_layout(entries=bind_group_entries)
        del bind_group_entries

        pipeline_layout = device.create_pipeline_layout(
            bind_group_layouts=[bind_group_layout]
        )
        del bind_group_layout

        pipeline = device.create_render_pipeline(
            layout=pipeline_layout,
            vertex={
                "module": shader,
                "buffers": [VERTEX_BUFFER_LAYOUT],
            },
            fragment={
                "module": shader,
                "targets": [{"format": output_texture_format}],
            },
            depth_stencil={"format": depth_texture_format},
            primitive={"topology": "triangle-strip"},
        )
        del pipeline_layout

        vertex_buffer = device.create_buffer_with_data(
            data=struct.pack("8f", -0.5, -0.5, -0.5, +0.5, +0.5, -0.5, +0.5, +0.5),
            usage=wgpu.BufferUsage.VERTEX,
        )

        indirect_buffer = device.create_buffer_with_data(
            # Used for both indexed and non-indexed draws.
            data=struct.pack("5i", 4, 1, 0, 0, 0),
            usage=wgpu.BufferUsage.INDIRECT,
        )

        index_buffer = device.create_buffer_with_data(
            data=struct.pack("4i", 0, 1, 2, 3),
            usage=wgpu.BufferUsage.INDEX,
        )

        data_buffer = device.create_buffer(size=2 * 4, usage="UNIFORM")
        bind_group = device.create_bind_group(
            layout=pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": data_buffer}},
            ],
        )
        del data_buffer

        output_texture = device.create_texture(
            size=[128, 128],
            format=output_texture_format,
            usage="RENDER_ATTACHMENT",
        )
        depth_texture = device.create_texture(
            size=[128, 128],
            format=depth_texture_format,
            usage="RENDER_ATTACHMENT",
        )
        occlusion_query_set = device.create_query_set(type="occlusion", count=2)
        timestamp_query_set = None
        if has_timestamps:
            timestamp_query_set = device.create_query_set(type="timestamp", count=2)

        render_pass_descriptor = {
            "color_attachments": [
                {
                    "clear_value": (0, 0, 0, 0),
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                    "view": output_texture.create_view(),
                }
            ],
            "depth_stencil_attachment": {
                "view": depth_texture.create_view(),
                "depth_clear_value": 0.1,
                "depth_load_op": wgpu.LoadOp.clear,
                "depth_store_op": wgpu.StoreOp.store,
            },
            "occlusion_query_set": occlusion_query_set,
        }
        if has_timestamps:
            render_pass_descriptor["timestamp_writes"] = {
                "query_set": timestamp_query_set,
                "beginning_of_pass_write_index": 0,
                "end_of_pass_write_index": 1,
            }

        del output_texture
        del depth_texture
        del occlusion_query_set
        del timestamp_query_set

        def do_draw(encoder):
            nonlocal pipeline, bind_group, vertex_buffer, index_buffer, indirect_buffer
            encoder.set_pipeline(pipeline)
            encoder.set_bind_group(0, bind_group)
            encoder.set_vertex_buffer(0, vertex_buffer)
            encoder.set_index_buffer(index_buffer, "uint32")
            encoder.draw(4, 1, 0, 0)
            encoder.draw_indexed(4, 1, 0, 0)
            encoder.draw_indirect(indirect_buffer, 0)
            encoder.draw_indexed_indirect(indirect_buffer, 0)
            pipeline = bind_group = vertex_buffer = index_buffer = indirect_buffer = (
                None
            )

        command_encoder = device.create_command_encoder()
        render_pass = command_encoder.begin_render_pass(**render_pass_descriptor)
        del render_pass_descriptor

        if use_render_bundle:
            render_bundle_encoder = device.create_render_bundle_encoder(
                color_formats=[output_texture_format],
                depth_stencil_format=depth_texture_format,
            )
            do_draw(render_bundle_encoder)

            render_bundle = render_bundle_encoder.finish()
            render_pass.execute_bundles([render_bundle])
        else:
            if has_statistics:
                statistics_query_set = create_statistics_query_set(
                    device,
                    count=2,
                    statistics=[PipelineStatisticName.FragmentShaderInvocations],
                )
                begin_pipeline_statistics_query(render_pass, statistics_query_set, 0)
                del statistics_query_set
            do_draw(render_pass)
            if has_statistics:
                end_pipeline_statistics_query(render_pass)

        render_pass.end()

        device.queue.submit([command_encoder.finish()])


def test_object_retention_in_compute():
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    desired_features = ["timestamp-query", "pipeline-statistics-query"]
    features = [x for x in desired_features if x in adapter.features]
    device = adapter.request_device_sync(required_features=features)
    has_timestamps = "timestamp-query" in features
    has_statistics = "pipeline-statistics-query" in features

    shader = device.create_shader_module(code=SHADER_SOURCE)

    bind_group_entries = BIND_GROUP_ENTRIES.copy()
    bind_group_layout = device.create_bind_group_layout(entries=bind_group_entries)
    del bind_group_entries

    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )
    del bind_group_layout

    pipeline = device.create_compute_pipeline(
        layout=pipeline_layout,
        compute={
            "module": shader,
        },
    )
    del pipeline_layout

    data_buffer = device.create_buffer(size=2 * 4, usage="UNIFORM")

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": data_buffer}},
        ],
    )
    del data_buffer

    timestamp_query_set = None
    if has_timestamps:
        timestamp_query_set = device.create_query_set(type="timestamp", count=2)

    compute_pass_descriptor = {}
    if has_timestamps:
        compute_pass_descriptor["timestamp_writes"] = {
            "query_set": timestamp_query_set,
            "beginning_of_pass_write_index": 0,
            "end_of_pass_write_index": 1,
        }

    del timestamp_query_set

    def setup_compute(encoder):
        indirect_buffer = device.create_buffer_with_data(
            # Used for both indexed and non-indexed draws.
            data=struct.pack("3i", 1, 1, 1),
            usage=wgpu.BufferUsage.INDIRECT,
        )
        nonlocal pipeline, bind_group
        encoder.set_pipeline(pipeline)
        encoder.set_bind_group(0, bind_group)
        encoder.dispatch_workgroups(1, 1, 1)
        encoder.dispatch_workgroups_indirect(indirect_buffer, 0)
        pipeline = bind_group = None

    command_encoder = device.create_command_encoder()
    compute_pass = command_encoder.begin_compute_pass(**compute_pass_descriptor)
    del compute_pass_descriptor

    if has_statistics:
        statistics_query_set = create_statistics_query_set(
            device,
            count=2,
            statistics=[PipelineStatisticName.ComputeShaderInvocations],
        )
        begin_pipeline_statistics_query(compute_pass, statistics_query_set, 0)
        del statistics_query_set
    setup_compute(compute_pass)
    if has_statistics:
        end_pipeline_statistics_query(compute_pass)

    compute_pass.end()
    device.queue.submit([command_encoder.finish()])


def test_clear_buffer():
    device = wgpu.utils.get_default_device()

    buffer = device.create_buffer(size=100, usage=wgpu.BufferUsage.COPY_DST)
    command_encoder = device.create_command_encoder()
    command_encoder.clear_buffer(buffer)
    del buffer
    device.queue.submit([command_encoder.finish()])


def test_copy_buffer_to_buffer():
    device = wgpu.utils.get_default_device()

    buffer1 = device.create_buffer(size=100, usage=wgpu.BufferUsage.COPY_SRC)
    buffer2 = device.create_buffer(size=100, usage=wgpu.BufferUsage.COPY_DST)
    command_encoder = device.create_command_encoder()
    command_encoder.copy_buffer_to_buffer(buffer1, 0, buffer2, 0, 100)
    del buffer1
    del buffer2
    device.queue.submit([command_encoder.finish()])


def test_copy_buffer_to_texture():
    device = wgpu.utils.get_default_device()

    texture = device.create_texture(size=[64, 64], format="r32uint", usage="COPY_DST")
    buffer = device.create_buffer(size=texture._nbytes, usage=wgpu.BufferUsage.COPY_SRC)
    command_encoder = device.create_command_encoder()
    command_encoder.copy_buffer_to_texture(
        {"buffer": buffer, "offset": 0, "bytes_per_row": 64 * 4, "rows_per_image": 64},
        {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
        (64, 64),
    )
    del buffer
    del texture
    device.queue.submit([command_encoder.finish()])


def test_copy_texture_to_buffer():
    device = wgpu.utils.get_default_device()

    texture = device.create_texture(size=[64, 64], format="r32uint", usage="COPY_SRC")
    buffer = device.create_buffer(size=texture._nbytes, usage=wgpu.BufferUsage.COPY_DST)
    command_encoder = device.create_command_encoder()
    command_encoder.copy_texture_to_buffer(
        {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
        {"buffer": buffer, "offset": 0, "bytes_per_row": 64 * 4, "rows_per_image": 64},
        (64, 64),
    )
    del buffer
    del texture
    device.queue.submit([command_encoder.finish()])


def test_copy_texture_to_texture():
    device = wgpu.utils.get_default_device()

    texture1 = device.create_texture(size=[64, 64], format="r32uint", usage="COPY_SRC")
    texture2 = device.create_texture(size=[64, 64], format="r32uint", usage="COPY_DST")
    command_encoder = device.create_command_encoder()
    command_encoder.copy_texture_to_texture(
        {"texture": texture1, "mip_level": 0, "origin": (0, 0, 0)},
        {"texture": texture2, "mip_level": 0, "origin": (0, 0, 0)},
        (64, 64),
    )
    del texture1
    del texture2
    device.queue.submit([command_encoder.finish()])


if __name__ == "__main__":
    test_object_retention_in_render()
    test_object_retention_in_compute()
    test_clear_buffer()
    test_copy_buffer_to_buffer()
    test_copy_buffer_to_texture()
    test_copy_texture_to_buffer()
    test_copy_texture_to_texture()
