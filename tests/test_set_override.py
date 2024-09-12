import numpy as np
import pytest

import wgpu.utils
from tests.testutils import can_use_wgpu_lib, run_tests
from wgpu import TextureFormat

if not can_use_wgpu_lib:
    pytest.skip("Skipping tests that need the wgpu lib", allow_module_level=True)


"""
This code is an amazingly slow way of adding together two 10-element arrays of 32-bit
integers defined by push constants and store them into an output buffer.


The source code assumes the topology is POINT-LIST, so that each call to vertexMain
corresponds with one call to fragmentMain. We draw exactly one point.
"""

SHADER_SOURCE = """
    @id(0) override a: i32 = 1;
    @id(1) override b: u32 = 2u;
    @id(2) override c: f32 = 3.0;
    @id(3) override d: bool = true;

    // Put the results here
    @group(0) @binding(0) var<storage, read_write> data: array<u32>;

    struct VertexOutput {
        @location(0) a: i32,
        @location(1) b: u32,
        @location(2) c: f32,
        @location(3) d: u32,
        @builtin(position) position: vec4f,
    }

    @vertex
    fn vertexMain(@builtin(vertex_index) index: u32) -> VertexOutput {
        return VertexOutput(a, b, c, u32(d), vec4f(0, 0, 0, 1));
    }

    @fragment
    fn fragmentMain(output: VertexOutput) -> @location(0) vec4f {
        data[0] = u32(output.a);
        data[1] = u32(output.b);
        data[2] = u32(output.c);
        data[3] = u32(output.d);
        data[4] = u32(a);
        data[5] = u32(b);
        data[6] = u32(c);
        data[7] = u32(d);
        return vec4f();
    }

    @compute @workgroup_size(1)
    fn computeMain() {
        data[0] = u32(a);
        data[1] = u32(b);
        data[2] = u32(c);
        data[3] = u32(d);
    }

"""

BIND_GROUP_ENTRIES = [
    {"binding": 0, "visibility": "FRAGMENT|COMPUTE", "buffer": {"type": "storage"}},
]


def test_override_constants():
    device = wgpu.utils.get_default_device()
    output_texture = device.create_texture(
        # Actual size is immaterial.  Could just be 1x1
        size=[128, 128],
        format=TextureFormat.rgba8unorm,
        usage="RENDER_ATTACHMENT|COPY_SRC",
    )
    shader = device.create_shader_module(code=SHADER_SOURCE)
    bind_group_layout = device.create_bind_group_layout(entries=BIND_GROUP_ENTRIES)
    render_pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout],
    )
    pipeline = device.create_render_pipeline(
        layout=render_pipeline_layout,
        vertex={
            "module": shader,
            "constants": {0: -2, "1": 10, "2": 20.0, "3": 23},
        },
        fragment={
            "module": shader,
            "targets": [{"format": output_texture.format}],
            "constants": {"a": -5, "b": 20, "c": 30.0, "d": True},
        },
        primitive={
            "topology": "point-list",
        },
    )
    render_pass_descriptor = {
        "color_attachments": [
            {
                "clear_value": (0, 0, 0, 0),  # only first value matters
                "load_op": "clear",
                "store_op": "store",
                "view": output_texture.create_view(),
            }
        ],
    }
    output_buffer = device.create_buffer(size=8 * 4, usage="STORAGE|COPY_SRC")
    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": output_buffer}},
        ],
    )

    encoder = device.create_command_encoder()
    this_pass = encoder.begin_render_pass(**render_pass_descriptor)
    this_pass.set_pipeline(pipeline)
    this_pass.set_bind_group(0, bind_group)
    this_pass.draw(1)
    this_pass.end()
    device.queue.submit([encoder.finish()])
    info_view = device.queue.read_buffer(output_buffer)
    result = np.frombuffer(info_view, dtype=np.uint32)
    print(result)


if __name__ == "__main__":
    run_tests(globals())
