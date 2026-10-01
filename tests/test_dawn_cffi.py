"""Tests for the cffi-based Dawn backend (``wgpu.backends.dawn`` with ``wgpu_dawn``).

Like ``test_dawn_backend.py``, these only run with ``WGPUPY_BACKEND=dawn``.
"""

import os

import numpy as np
import pytest

import wgpu


if os.getenv("WGPUPY_BACKEND", "").lower() != "dawn":
    pytest.skip(
        "Dawn backend not selected (WGPUPY_BACKEND=dawn)", allow_module_level=True
    )

dawn = pytest.importorskip("wgpu.backends.dawn")


@pytest.fixture(scope="module")
def device():
    return wgpu.gpu.request_adapter_sync().request_device_sync()


def test_enumerate_adapters():
    adapters = wgpu.gpu.enumerate_adapters_sync()
    assert adapters
    assert all(isinstance(a, dawn.GPUAdapter) for a in adapters)
    keys = [(a.summary, a.info["backend_type"]) for a in adapters]
    assert len(set(keys)) == len(keys)
    for a in adapters:
        assert a.info["backend_type"] != "Null"
        # Like the wgpu-native backend, the vendor is the driver name on Vulkan,
        # so that e.g. lavapipe can be detected with vendor == "llvmpipe".
        if a.info["backend_type"] == "Vulkan" and "llvmpipe" in a.info["device"]:
            assert a.info["vendor"] == "llvmpipe"
    # Each adapter can create a device
    adapters[-1].request_device_sync()


def test_encoder_errors_raise_at_finish(device):
    texture = device.create_texture(
        size=(4, 4, 1), format="rgba8unorm", usage=wgpu.TextureUsage.RENDER_ATTACHMENT
    )
    encoder = device.create_command_encoder()
    rpass = encoder.begin_render_pass(
        color_attachments=[
            {"view": texture.create_view(), "load_op": "clear", "store_op": "store"}
        ]
    )
    rpass.draw(3)  # no pipeline set: the error is deferred to finish()
    rpass.end()
    with pytest.raises(wgpu.GPUValidationError):
        encoder.finish()
    # And the next call is fine again
    device.create_command_encoder().finish()


DYNAMIC_SHADER = """
@group(0) @binding(0) var<uniform> value: vec4<u32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(1)
fn main() {
    out[value.y] = value.x;
}
"""


@pytest.mark.parametrize("how", ["list", "numpy", "start_length"])
def test_set_bind_group_dynamic_offsets(device, how):
    # Two uniform values, 256 bytes apart (minUniformBufferOffsetAlignment)
    data = np.zeros(128, np.uint32)
    data[0:2] = 10, 0
    data[64:66] = 20, 1
    ubuf = device.create_buffer_with_data(data=data, usage=wgpu.BufferUsage.UNIFORM)
    sbuf = device.create_buffer(
        size=8, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
    )
    bgl = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": "uniform", "has_dynamic_offset": True},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": "storage"},
            },
        ]
    )
    bind_group = device.create_bind_group(
        layout=bgl,
        entries=[
            {"binding": 0, "resource": {"buffer": ubuf, "offset": 0, "size": 16}},
            {"binding": 1, "resource": {"buffer": sbuf}},
        ],
    )
    pipeline = device.create_compute_pipeline(
        layout=device.create_pipeline_layout(bind_group_layouts=[bgl]),
        compute={"module": device.create_shader_module(code=DYNAMIC_SHADER)},
    )
    encoder = device.create_command_encoder()
    cpass = encoder.begin_compute_pass()
    cpass.set_pipeline(pipeline)
    for offset in (0, 256):
        if how == "list":
            cpass.set_bind_group(0, bind_group, [offset])
        elif how == "numpy":
            cpass.set_bind_group(0, bind_group, np.array([offset], np.uint32))
        else:
            cpass.set_bind_group(0, bind_group, [99, offset, 99], 1, 1)
        cpass.dispatch_workgroups(1)
    cpass.end()
    device.queue.submit([encoder.finish()])
    out = np.frombuffer(device.queue.read_buffer(sbuf), np.uint32)
    assert list(out) == [10, 20]


def test_render_bundle_with_depth_format(device):
    bundle_encoder = device.create_render_bundle_encoder(
        color_formats=["rgba8unorm"], depth_stencil_format="depth24plus"
    )
    bundle = bundle_encoder.finish()
    assert isinstance(bundle, dawn.GPURenderBundle)
