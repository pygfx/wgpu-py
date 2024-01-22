import os
import base64
import shutil
import ctypes
import sys
import tempfile

import wgpu.utils
import wgpu.backends.wgpu_native
import numpy as np

from testutils import run_tests, can_use_wgpu_lib, is_ci
from pytest import mark, raises


is_win = sys.platform.startswith("win")


def test_get_wgpu_version():
    version = wgpu.backends.wgpu_native.__version__
    commit_sha = wgpu.backends.wgpu_native.__commit_sha__
    version_info = wgpu.backends.wgpu_native.version_info

    assert isinstance(version, str)
    assert len(version) > 1

    assert isinstance(version_info, tuple)
    assert all(isinstance(i, int) for i in version_info)
    assert len(version_info) == 4

    assert isinstance(commit_sha, str)
    assert len(commit_sha) > 0


def test_override_wgpu_lib_path():
    # Current version
    try:
        old_path = wgpu.backends.wgpu_native.lib_path
    except RuntimeError:
        old_path = None

    # Change it
    old_env_var = os.environ.get("WGPU_LIB_PATH", None)
    os.environ["WGPU_LIB_PATH"] = "foo/bar"

    # Check
    assert wgpu.backends.wgpu_native._ffi.get_wgpu_lib_path() == "foo/bar"

    # Change it back
    if old_env_var is None:
        os.environ.pop("WGPU_LIB_PATH")
    else:
        os.environ["WGPU_LIB_PATH"] = old_env_var

    # Still the same as before?
    try:
        path = wgpu.backends.wgpu_native._ffi.get_wgpu_lib_path()
    except RuntimeError:
        path = None
    assert path == old_path


def test_tuple_from_tuple_or_dict():
    func = wgpu.backends.wgpu_native._api._tuple_from_tuple_or_dict

    assert func([1, 2, 3], ("x", "y", "z")) == (1, 2, 3)
    assert func({"y": 2, "z": 3, "x": 1}, ("x", "y", "z")) == (1, 2, 3)
    assert func((10, 20), ("width", "height")) == (10, 20)
    assert func({"width": 10, "height": 20}, ("width", "height")) == (10, 20)

    with raises(TypeError):
        func("not tuple/dict", ("x", "y"))
    with raises(ValueError):
        func([1], ("x", "y"))
    with raises(ValueError):
        func([1, 2, 3], ("x", "y"))
    with raises(ValueError):
        assert func({"x": 1}, ("x", "y"))


compute_shader_wgsl = """
@group(0)
@binding(0)
var<storage,read_write> out1: array<i32>;

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) index: vec3<u32>) {
    let i: u32 = index.x;
    out1[i] = 1000 + i32(i);
}
"""

compute_shader_glsl = """
#version 450 core

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding=0) buffer BufferBinding { int out1[]; };

void main() {
    uvec3 index = gl_GlobalInvocationID;
    uint i = index.x;
    out1[i] = (1000 + int(i));
    return;
}
"""

compute_shader_spirv = base64.decodebytes(
    """
AwIjBwAAAQAcAAAAHAAAAAAAAAARAAIAAQAAAAoACwBTUFZfS0hSX3N0b3JhZ2VfYnVmZmVyX3N0
b3JhZ2VfY2xhc3MAAAAACwAGAAEAAABHTFNMLnN0ZC40NTAAAAAADgADAAAAAAABAAAADwAGAAUA
AAAQAAAAbWFpbgAAAAANAAAAEAAGABAAAAARAAAAAQAAAAEAAAABAAAABQAEAAkAAABvdXQxAAAA
AAUABAANAAAAaW5kZXgAAAAFAAQAEAAAAG1haW4AAAAARwAEAAQAAAAGAAAABAAAAEcABAAJAAAA
IgAAAAAAAABHAAQACQAAACEAAAAAAAAARwADAAoAAAACAAAASAAFAAoAAAAAAAAAIwAAAAAAAABH
AAQADQAAAAsAAAAcAAAAEwACAAIAAAAVAAQAAwAAACAAAAABAAAAHQADAAQAAAADAAAAFQAEAAYA
AAAgAAAAAAAAABcABAAFAAAABgAAAAMAAAArAAQAAwAAAAcAAAAAAAAAKwAEAAMAAAAIAAAAAQAA
AB4AAwAKAAAABAAAACAABAALAAAADAAAAAoAAAA7AAQACwAAAAkAAAAMAAAAIAAEAA4AAAABAAAA
BQAAADsABAAOAAAADQAAAAEAAAAhAAMAEQAAAAIAAAAgAAQAEgAAAAwAAAAEAAAAKwAEAAYAAAAT
AAAAAAAAACsABAADAAAAFQAAAOgDAAAgAAQAGAAAAAwAAAADAAAANgAFAAIAAAAQAAAAAAAAABEA
AAD4AAIADAAAAD0ABAAFAAAADwAAAA0AAABBAAUAEgAAABQAAAAJAAAAEwAAAPkAAgAWAAAA+AAC
ABYAAABRAAUABgAAABcAAAAPAAAAAAAAAHwABAADAAAAGQAAABcAAACAAAUAAwAAABoAAAAVAAAA
GQAAAEEABQAYAAAAGwAAABQAAAAXAAAAPgADABsAAAAaAAAA/QABADgAAQA=
""".encode()
)


def run_compute_shader(device, shader):
    """Minimal compute setup for the above shaders."""
    n = 16
    buffer = device.create_buffer(
        size=n * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
    )

    # Setup layout and bindings
    binding_layouts = [
        {
            "binding": 0,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {"type": wgpu.BufferBindingType.storage},
        },
    ]
    bindings = [
        {
            "binding": 0,
            "resource": {"buffer": buffer, "offset": 0, "size": buffer.size},
        },
    ]

    # Put everything together
    bind_group_layout = device.create_bind_group_layout(entries=binding_layouts)
    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )
    bind_group = device.create_bind_group(layout=bind_group_layout, entries=bindings)

    # Create and run the pipeline
    compute_pipeline = device.create_compute_pipeline(
        layout=pipeline_layout,
        compute={"module": shader, "entry_point": "main"},
    )
    command_encoder = device.create_command_encoder()
    compute_pass = command_encoder.begin_compute_pass()
    compute_pass.set_pipeline(compute_pipeline)
    compute_pass.set_bind_group(0, bind_group, [], 0, 999999)
    compute_pass.dispatch_workgroups(n, 1, 1)  # x y z
    compute_pass.end()
    device.queue.submit([command_encoder.finish()])

    # Read result
    out = device.queue.read_buffer(buffer).cast("i")
    result = out.tolist()
    assert result == [1000 + i for i in range(n)]


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_compute_shader_wgsl():
    device = wgpu.utils.get_default_device()

    code = compute_shader_wgsl
    assert isinstance(code, str)

    shader = device.create_shader_module(code=code)
    assert shader.get_compilation_info() == []

    run_compute_shader(device, shader)


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_compute_shader_glsl():
    device = wgpu.utils.get_default_device()

    code = compute_shader_glsl
    assert isinstance(code, str)

    shader = device.create_shader_module(label="simple comp", code=code)
    assert shader.get_compilation_info() == []

    run_compute_shader(device, shader)


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
@mark.skipif(is_ci and is_win, reason="Cannot use SpirV shader on dx12")
def test_compute_shader_spirv():
    device = wgpu.utils.get_default_device()

    code = compute_shader_spirv
    assert isinstance(code, bytes)

    shader = device.create_shader_module(code=code)
    assert shader.get_compilation_info() == []

    run_compute_shader(device, shader)


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_compute_shader_invalid():
    device = wgpu.utils.get_default_device()

    code4 = type("CodeObject", (object,), {})

    with raises(TypeError):
        device.create_shader_module(code=code4)
    with raises(TypeError):
        device.create_shader_module(code={"not", "a", "shader"})
    with raises(ValueError):
        device.create_shader_module(code=b"bytes but no SpirV magic number")


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_logging():
    # Do *something* while we set the log level low
    device = wgpu.utils.get_default_device()

    wgpu.logger.setLevel("DEBUG")

    device.create_shader_module(code=compute_shader_wgsl)

    wgpu.logger.setLevel("WARNING")

    # yeah, would be nice to be able to capture the logs. But if we don't crash
    # and see from the coverage that we touched the logger integration code,
    # we're doing pretty good ...
    # (capsys does not work because it logs to the raw stderr)


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_wgpu_native_tracer():
    tempdir = os.path.join(tempfile.gettempdir(), "wgpu-tracer-test")
    adapter = wgpu.utils.get_default_device().adapter

    # Make empty
    shutil.rmtree(tempdir, ignore_errors=True)
    assert not os.path.isdir(tempdir)

    # Works!
    wgpu.backends.wgpu_native.request_device_tracing(adapter, tempdir)
    assert os.path.isdir(tempdir)

    # Make dir not empty
    with open(os.path.join(tempdir, "stub.txt"), "wb"):
        pass

    # Still works, but produces warning
    wgpu.backends.wgpu_native.request_device_tracing(adapter, tempdir)


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_wgpu_native_enumerate_adapters():
    # Get all available adapters
    adapters = wgpu.backends.wgpu_native.enumerate_adapters()
    assert len(adapters) > 0

    # Check that we can get a device from each adapter
    for adapter in adapters:
        d = adapter.request_device()
        assert isinstance(d, wgpu.backends.wgpu_native.GPUDevice)


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_adapter_destroy():
    adapter = wgpu.gpu.request_adapter(power_preference="high-performance")
    assert adapter._internal is not None
    adapter.__del__()
    assert adapter._internal is None


def test_get_memoryview_and_address():
    get_memoryview_and_address = (
        wgpu.backends.wgpu_native._helpers.get_memoryview_and_address
    )

    data = b"bytes are readonly, but we can map it. Don't abuse this :)"
    m, address = get_memoryview_and_address(data)
    assert m.nbytes == len(data)
    assert address > 0

    data = bytearray(b"A bytearray works too")
    m, address = get_memoryview_and_address(data)
    assert m.nbytes == len(data)
    assert address > 0

    data = (ctypes.c_float * 100)()
    m, address = get_memoryview_and_address(data)
    assert m.nbytes == ctypes.sizeof(data)
    assert address > 0

    data = np.array([1, 2, 3, 4])
    m, address = get_memoryview_and_address(data)
    assert m.nbytes == data.nbytes
    assert address > 0

    data = np.array([1, 2, 3, 4])
    data.flags.writeable = False
    m, address = get_memoryview_and_address(data)
    assert m.nbytes == data.nbytes
    assert address > 0


if __name__ == "__main__":
    run_tests(globals())
