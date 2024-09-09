import numpy as np
import pytest

import wgpu.utils
from tests.testutils import can_use_wgpu_lib, run_tests
from wgpu import TextureFormat

if not can_use_wgpu_lib:
    pytest.skip("Skipping tests that need the wgpu lib", allow_module_level=True)

COUNT = 10

SHADER_SOURCE = (
    f"""
    const COUNT = {COUNT}u;
"""
    """
    @group(0) @binding(0) var<storage, read_write> data: array<u32, COUNT>;

    struct PushConstants {
        values1: array<u32, COUNT>,
        values2: array<u32, COUNT>,
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


def run_test(device):
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

    vertex_call_buffer = device.create_buffer(size=COUNT * 4, usage="STORAGE|COPY_SRC")

    bind_group = device.create_bind_group(
        label=f"Bind Group for Faces",
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": vertex_call_buffer}},
        ],
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

    buffer1 = np.random.randint(0, 1_000_000, COUNT, dtype=np.uint32)
    buffer2 = np.random.randint(0, 1_000_000, COUNT, dtype=np.uint32)

    encoder = device.create_command_encoder()
    this_pass = encoder.begin_render_pass(**render_pass_descriptor)
    this_pass.set_pipeline(pipeline)
    this_pass.set_bind_group(0, bind_group)
    this_pass.set_push_constants("VERTEX", 0, COUNT * 4, buffer1)
    this_pass.set_push_constants("FRAGMENT", COUNT * 4, COUNT * 4, buffer2)
    this_pass.draw(COUNT)
    this_pass.end()
    device.queue.submit([encoder.finish()])
    info_view = device.queue.read_buffer(vertex_call_buffer)
    info = np.frombuffer(info_view, dtype=np.uint32)
    assert all(buffer1 + buffer2 == info)


def get_device():
    adapter = wgpu.gpu.request_adapter(power_preference="high-performance")
    device = adapter.request_device(
        required_features=["push-constants"],
        required_limits={"max_push_constant_size": 128},
    )
    return device


def test_me():
    device = get_device()
    run_test(device)


if __name__ == "__main__":
    run_tests(globals())
