import numpy as np
import pytest

import wgpu.utils
from tests.testutils import can_use_wgpu_lib, run_tests
from wgpu import TextureFormat
from wgpu.backends.wgpu_native.extras import create_pipeline_layout, set_push_constants

if not can_use_wgpu_lib:
    pytest.skip("Skipping tests that need the wgpu lib", allow_module_level=True)


"""
This code is an amazingly slow way of adding together two 10-element arrays of 32-bit
integers defined by push constants and store them into an output buffer.

The first number of the addition is purposely pulled using the vertex stage, and the
second number from the fragment stage, so that we can ensure that we are correctly
using stage-separated push constants correctly.

The source code assumes the topology is POINT-LIST, so that each call to vertexMain
corresponds with one call to fragmentMain.
"""
COUNT = 10

SHADER_SOURCE = (
    f"""
    const COUNT = {COUNT}u;
"""
    """
    // Put the results here
    @group(0) @binding(0) var<storage, read_write> data: array<u32, COUNT>;

    struct PushConstants {
        values1: array<u32, COUNT>, // VERTEX constants
        values2: array<u32, COUNT>, // FRAGMENT constants
    }
    var<push_constant> push_constants: PushConstants;

    struct VertexOutput {
        @location(0) index: u32,
        @location(1) value: u32,
        @builtin(position) position: vec4f,
    }

    @vertex
    fn vertexMain(
        @builtin(vertex_index) index: u32,
    ) -> VertexOutput {
        return VertexOutput(index, push_constants.values1[index], vec4f(0, 0, 0, 1));
    }

    @fragment
    fn fragmentMain(@location(0) index: u32,
                    @location(1) value: u32
    ) -> @location(0) vec4f {
        data[index] = value + push_constants.values2[index];
        return vec4f();
    }
"""
)

BIND_GROUP_ENTRIES = [
    {"binding": 0, "visibility": "FRAGMENT", "buffer": {"type": "storage"}},
]


def setup_pipeline():
    adapter = wgpu.gpu.request_adapter(power_preference="high-performance")
    device = adapter.request_device(
        required_features=["push-constants"],
        required_limits={"max-push-constant-size": 128},
    )
    output_texture = device.create_texture(
        # Actual size is immaterial.  Could just be 1x1
        size=[128, 128],
        format=TextureFormat.rgba8unorm,
        usage="RENDER_ATTACHMENT|COPY_SRC",
    )
    shader = device.create_shader_module(code=SHADER_SOURCE)
    bind_group_layout = device.create_bind_group_layout(entries=BIND_GROUP_ENTRIES)
    render_pipeline_layout = create_pipeline_layout(
        device,
        bind_group_layouts=[bind_group_layout],
        push_constant_layouts=[
            {"visibility": "VERTEX", "start": 0, "end": COUNT * 4},
            {"visibility": "FRAGMENT", "start": COUNT * 4, "end": COUNT * 4 * 2},
        ],
    )
    pipeline = device.create_render_pipeline(
        layout=render_pipeline_layout,
        vertex={
            "module": shader,
            "entry_point": "vertexMain",
        },
        fragment={
            "module": shader,
            "entry_point": "fragmentMain",
            "targets": [{"format": output_texture.format}],
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

    return device, pipeline, render_pass_descriptor


def test_normal_push_constants():
    device, pipeline, render_pass_descriptor = setup_pipeline()
    vertex_call_buffer = device.create_buffer(size=COUNT * 4, usage="STORAGE|COPY_SRC")
    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": vertex_call_buffer}},
        ],
    )

    encoder = device.create_command_encoder()
    this_pass = encoder.begin_render_pass(**render_pass_descriptor)
    this_pass.set_pipeline(pipeline)
    this_pass.set_bind_group(0, bind_group)

    buffer = np.random.randint(0, 1_000_000, size=(2 * COUNT), dtype=np.uint32)
    set_push_constants(this_pass, "VERTEX", 0, COUNT * 4, buffer)
    set_push_constants(this_pass, "FRAGMENT", COUNT * 4, COUNT * 4, buffer, COUNT * 4)
    this_pass.draw(COUNT)
    this_pass.end()
    device.queue.submit([encoder.finish()])
    info_view = device.queue.read_buffer(vertex_call_buffer)
    result = np.frombuffer(info_view, dtype=np.uint32)
    expected_result = buffer[0:COUNT] + buffer[COUNT:]
    assert all(result == expected_result)


def test_bad_set_push_constants():
    device, pipeline, render_pass_descriptor = setup_pipeline()
    encoder = device.create_command_encoder()
    this_pass = encoder.begin_render_pass(**render_pass_descriptor)

    def zeros(n):
        return np.zeros(n, dtype=np.uint32)

    with pytest.raises(ValueError):
        # Buffer is to short
        set_push_constants(this_pass, "VERTEX", 0, COUNT * 4, zeros(COUNT - 1))

    with pytest.raises(ValueError):
        # Buffer is to short
        set_push_constants(this_pass, "VERTEX", 0, COUNT * 4, zeros(COUNT + 1), 8)


if __name__ == "__main__":
    run_tests(globals())
