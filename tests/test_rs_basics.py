import os
import base64
import shutil
import random
import ctypes
import sys
import tempfile

import wgpu.utils
import wgpu.backends.rs
import numpy as np

from testutils import run_tests, can_use_wgpu_lib, is_ci, iters_equal
from pytest import mark, raises


is_win = sys.platform.startswith("win")


def test_get_wgpu_version():
    version = wgpu.backends.rs.__version__
    commit_sha = wgpu.backends.rs.__commit_sha__
    version_info = wgpu.backends.rs.version_info

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
        old_path = wgpu.backends.rs_ffi.get_wgpu_lib_path()
    except RuntimeError:
        old_path = None

    # Change it
    old_env_var = os.environ.get("WGPU_LIB_PATH", None)
    os.environ["WGPU_LIB_PATH"] = "foo/bar"

    # Check
    assert wgpu.backends.rs_ffi.get_wgpu_lib_path() == "foo/bar"

    # Change it back
    if old_env_var is None:
        os.environ.pop("WGPU_LIB_PATH")
    else:
        os.environ["WGPU_LIB_PATH"] = old_env_var

    # Still the same as before?
    try:
        path = wgpu.backends.rs_ffi.get_wgpu_lib_path()
    except RuntimeError:
        path = None
    assert path == old_path


def test_tuple_from_tuple_or_dict():
    func = wgpu.backends.rs._tuple_from_tuple_or_dict

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
    out1[i] = i32(i);
}
"""

compute_shader_spirv = base64.decodebytes(
    """
AwIjBwADAQAAAAAAFgAAAAAAAAARAAIAAQAAAA4AAwAAAAAAAAAAAA8ABgAFAAAAAQAAAG1haW4A
AAAACAAAABAABgABAAAAEQAAAAEAAAABAAAAAQAAAAUABAABAAAAbWFpbgAAAAAFAAQACAAAAGlu
ZGV4AAAABQADAAwAAABvdXQABQADAA0AAAAwAAAARwAEAAgAAAALAAAAHAAAAEcABAAJAAAABgAA
AAQAAABIAAUACgAAAAAAAAAjAAAAAAAAAEcAAwAKAAAAAwAAAEcABAAMAAAAIgAAAAAAAABHAAQA
DAAAACEAAAAAAAAAEwACAAIAAAAhAAMAAwAAAAIAAAAVAAQABQAAACAAAAABAAAAFwAEAAYAAAAF
AAAAAwAAACAABAAHAAAAAQAAAAYAAAA7AAQABwAAAAgAAAABAAAAHQADAAkAAAAFAAAAHgADAAoA
AAAJAAAAIAAEAAsAAAACAAAACgAAADsABAALAAAADAAAAAIAAAArAAQABQAAAA0AAAAAAAAAIAAE
AA4AAAACAAAABQAAACAABAAQAAAAAQAAAAUAAAAgAAQAEwAAAAEAAAAFAAAANgAFAAIAAAABAAAA
AAAAAAMAAAD4AAIABAAAAEEABQAQAAAAEQAAAAgAAAANAAAAPQAEAAUAAAASAAAAEQAAAEEABgAO
AAAADwAAAAwAAAANAAAAEgAAAEEABQATAAAAFAAAAAgAAAANAAAAPQAEAAUAAAAVAAAAFAAAAD4A
AwAPAAAAFQAAAP0AAQA4AAEA
""".encode()
)


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
def test_rs_tracer():
    tempdir = os.path.join(tempfile.gettempdir(), "wgpu-tracer-test")
    adapter = wgpu.utils.get_default_device().adapter

    # Make empty
    shutil.rmtree(tempdir, ignore_errors=True)
    assert not os.path.isdir(tempdir)

    # Works!
    adapter.request_device_tracing(tempdir)
    assert os.path.isdir(tempdir)

    # Make dir not empty
    with open(os.path.join(tempdir, "stub.txt"), "wb"):
        pass

    # Still works, but produces warning
    adapter.request_device_tracing(tempdir)


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
@mark.skipif(is_ci and is_win, reason="Cannot use SpirV shader on dx12")
def test_shader_module_creation_spirv():
    device = wgpu.utils.get_default_device()

    code1 = compute_shader_spirv
    assert isinstance(code1, bytes)
    code4 = type("CodeObject", (object,), {})

    m1 = device.create_shader_module(code=code1)
    assert m1.get_compilation_info() == []

    with raises(TypeError):
        device.create_shader_module(code=code4)
    with raises(TypeError):
        device.create_shader_module(code={"not", "a", "shader"})
    with raises(ValueError):
        device.create_shader_module(code=b"bytes but no SpirV magic number")


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_adapter_destroy():
    adapter = wgpu.request_adapter(canvas=None, power_preference="high-performance")
    assert adapter._internal is not None
    adapter.__del__()
    assert adapter._internal is None


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_buffer_init1():
    # Initializing a buffer with data

    device = wgpu.utils.get_default_device()
    data1 = b"abcdefghijkl"

    # Create buffer. COPY_SRC is needed to read the buffer via the queue.
    buf = device.create_buffer_with_data(data=data1, usage=wgpu.BufferUsage.COPY_SRC)

    # Download from buffer to CPU
    data2 = device.queue.read_buffer(buf)
    assert data1 == data2

    # ---  also read via mapped data

    # Create buffer. MAP_READ is needed to read the buffer via the queue.
    buf = device.create_buffer_with_data(data=data1, usage=wgpu.BufferUsage.MAP_READ)

    wgpu.backends.rs.libf.wgpuDevicePoll(
        buf._device._internal, True, wgpu.backends.rs.ffi.NULL
    )

    # Download from buffer to CPU
    buf.map(wgpu.MapMode.READ)
    wgpu.backends.rs.libf.wgpuDevicePoll(
        buf._device._internal, True, wgpu.backends.rs.ffi.NULL
    )

    data2 = buf.read_mapped()
    buf.unmap()
    print(data2.tobytes())
    assert data1 == data2


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_buffer_init2():
    # Initializing a buffer as mapped, to directly set the data

    device = wgpu.utils.get_default_device()
    data1 = b"abcdefghijkl"

    # Create buffer.
    buf = device.create_buffer(
        size=len(data1), usage=wgpu.BufferUsage.COPY_SRC, mapped_at_creation=True
    )
    buf.write_mapped(data1)
    buf.unmap()

    # Download from buffer to CPU
    data2 = device.queue.read_buffer(buf)
    assert data1 == data2

    # --- also read via mapped data

    # Create buffer.
    buf = device.create_buffer(
        size=len(data1), usage=wgpu.BufferUsage.MAP_READ, mapped_at_creation=True
    )
    buf.write_mapped(data1)
    buf.unmap()

    # Download from buffer to CPU
    buf.map("read")
    data2 = buf.read_mapped()
    buf.unmap()
    print(data2.tobytes())
    assert data1 == data2


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_buffer_init3():
    # Initializing an empty buffer, then writing to it, then reading back

    device = wgpu.utils.get_default_device()
    data1 = b"abcdefghijkl"

    # Option 1: write via queue (i.e. temp buffer), read via queue

    # Create buffer
    buf = device.create_buffer(
        size=len(data1), usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
    )

    # Write data to it
    device.queue.write_buffer(buf, 0, data1)

    # Download from buffer to CPU
    data2 = device.queue.read_buffer(buf)
    assert data1 == data2

    # Option 2: Write via mapped data, read via queue

    # Create buffer
    buf = device.create_buffer(
        size=len(data1), usage=wgpu.BufferUsage.MAP_WRITE | wgpu.BufferUsage.COPY_SRC
    )

    # Write data to it
    buf.map("write")
    buf.write_mapped(data1)
    buf.unmap()

    # Download from buffer to CPU
    data2 = device.queue.read_buffer(buf)
    assert data1 == data2

    # Option 3: Write via queue, read via mapped data

    buf = device.create_buffer(
        size=len(data1), usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST
    )

    # Write data to it
    device.queue.write_buffer(buf, 0, data1)

    # Download from buffer to CPU
    buf.map("read")
    data2 = buf.read_mapped()
    buf.unmap()
    assert data1 == data2

    # Option 4: Write via mapped data, read via mapped data

    # Not actually an option
    with raises(wgpu.GPUValidationError):
        buf = device.create_buffer(
            size=len(data1),
            usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.MAP_WRITE,
        )


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_consequitive_writes1():
    # The inefficient way

    device = wgpu.utils.get_default_device()

    # Create buffer
    buf = device.create_buffer(
        size=32, usage=wgpu.BufferUsage.MAP_WRITE | wgpu.BufferUsage.COPY_SRC
    )

    # Write in parts
    for i in range(4):
        buf.map("write")
        buf.write_mapped(f"{i+1}".encode() * 8, i * 8)
        buf.unmap()

    # Download from buffer to CPU
    data = device.queue.read_buffer(buf)
    assert data == b"11111111222222223333333344444444"

    # Also in parts
    for i in range(4):
        data = device.queue.read_buffer(buf, i * 8, size=8)
        assert data == f"{i+1}".encode() * 8


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_consequitive_writes2():
    # The efficient way

    device = wgpu.utils.get_default_device()

    # Create buffer
    buf = device.create_buffer(
        size=32, usage=wgpu.BufferUsage.MAP_WRITE | wgpu.BufferUsage.COPY_SRC
    )

    # Write in parts
    buf.map("write")
    for i in range(4):
        buf.write_mapped(f"{i+1}".encode() * 8, i * 8)
    buf.unmap()

    # Download from buffer to CPU
    data = device.queue.read_buffer(buf)
    assert data == b"11111111222222223333333344444444"

    # Also in parts
    for i in range(4):
        data = device.queue.read_buffer(buf, i * 8, size=8)
        assert data == f"{i+1}".encode() * 8


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_consequitive_reads():
    device = wgpu.utils.get_default_device()

    # Create buffer
    buf = device.create_buffer(
        size=32, usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST
    )

    # Write using the queue. Do in parts, to touch those offsets too
    for i in range(4):
        device.queue.write_buffer(buf, i * 8, f"{i+1}".encode() * 8)

    # Read in parts, the inefficient way
    for i in range(4):
        buf.map("read")
        data = buf.read_mapped(i * 8, 8)
        assert data == f"{i+1}".encode() * 8
        buf.unmap()

    # Read in parts, the efficient way
    buf.map("read")
    for i in range(4):
        data = buf.read_mapped(i * 8, 8)
        assert data == f"{i+1}".encode() * 8
    buf.unmap()


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_buffer_mapping_fails():
    device = wgpu.utils.get_default_device()
    data = b"12345678"

    # Create buffer
    buf = device.create_buffer(
        size=32, usage=wgpu.BufferUsage.MAP_WRITE | wgpu.BufferUsage.COPY_SRC
    )

    with raises(RuntimeError):
        buf.write_mapped(data)  # Not mapped
    with raises(RuntimeError):
        buf.read_mapped()  # Not mapped

    with raises(ValueError):
        buf.map("boo")  # Invalid map mode

    buf.map("write", 0, 28)

    with raises(RuntimeError):
        buf.map("write")  # Cannot map twice

    with raises(RuntimeError):
        buf.map("read")  # Cannot map twice

    with raises(RuntimeError):
        buf.read_mapped()  # Not mapped in read mode

    # Ok
    buf.write_mapped(data)
    buf.write_mapped(data, 0)
    buf.write_mapped(data, 8)
    buf.write_mapped(data, 16)

    # Fail
    with raises(ValueError):
        buf.write_mapped(data, -1)  # not neg
    with raises(ValueError):
        buf.write_mapped(data, -8)  # not neg
    with raises(ValueError):
        buf.write_mapped(data, 6)  # not multilpe of eight

    # Ok
    buf.write_mapped(b"1" * 4)
    buf.write_mapped(b"1" * 8)
    buf.write_mapped(b"1" * 28)
    buf.write_mapped(b"1" * 12, 0)
    buf.write_mapped(b"1" * 12, 8)

    with raises(ValueError):
        buf.write_mapped(b"")  # not empty
    with raises(ValueError):
        buf.write_mapped(b"1" * 64)  # too large for buffer
    with raises(ValueError):
        buf.write_mapped(b"1" * 32)  # too large for mapped range
    with raises(ValueError):
        buf.write_mapped(b"1" * 3)  # not multiple of 4
    with raises(ValueError):
        buf.write_mapped(b"1" * 6)  # not multiple of 4
    with raises(ValueError):
        buf.write_mapped(b"1" * 9)  # not multiple of 4

    # Can unmap multiple times though!
    buf.unmap()

    with raises(RuntimeError):
        buf.unmap()  # Cannot unmap when not mapped

    # Create buffer in read mode ...

    buf = device.create_buffer(
        size=32, usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST
    )

    with raises(RuntimeError):
        buf.write_mapped(data)  # not mapped

    buf.map("read", 8, 20)

    with raises(RuntimeError):
        buf.map("read")  # Cannot map twice

    with raises(RuntimeError):
        buf.map("write")  # Cannot map twice

    with raises(RuntimeError):
        buf.write_mapped(data)  # not mapped in write mode

    # Ok
    assert len(buf.read_mapped()) == 20

    # Fail
    with raises(ValueError):
        buf.read_mapped(0, 64)  # read beyond buffer size
    with raises(ValueError):
        buf.read_mapped(0, 32)  # read beyond mapped range

    buf.unmap()

    with raises(RuntimeError):
        buf.unmap()  # Cannot unmap when not mapped


def test_buffer_read_no_copy():
    device = wgpu.utils.get_default_device()
    data1 = b"12345678" * 2

    buf = device.create_buffer(
        size=len(data1), usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST
    )

    # Write data to it
    device.queue.write_buffer(buf, 0, data1)

    # Download from buffer to CPU
    buf.map("read")
    data2 = buf.read_mapped(copy=False)
    data3 = buf.read_mapped(0, 8, copy=False)
    data4 = buf.read_mapped(8, 8, copy=False)

    assert data2 == data1
    assert data3 == data1[:8]
    assert data4 == data1[8:]

    # Can access the arrays
    _ = data2[0], data3[0], data4[0]

    # But cannot write to memory intended for reading
    if sys.version_info >= (3, 8):  # no memoryview.toreadonly on 3.7 and below
        with raises(TypeError):
            data2[0] = 1
        with raises(TypeError):
            data3[0] = 1
        with raises(TypeError):
            data4[0] = 1

    buf.unmap()

    # The memoryview is invalidated when the buffer unmapped.
    # Note that this unfortunately does *not* hold for views on these arrays.
    with raises(ValueError):
        data2[0]
    with raises(ValueError):
        data3[0]
    with raises(ValueError):
        data4[0]

    with raises(ValueError):
        data2[0] = 1
    with raises(ValueError):
        data3[0] = 1
    with raises(ValueError):
        data4[0] = 1


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_clear_buffer():
    data0 = b"111111112222222233333333"
    data1 = b"111111110000000000003333"
    data2 = b"111100000000000000000000"
    data3 = b"000000000000000000000000"

    # Prep
    device = wgpu.utils.get_default_device()
    buf = device.create_buffer(
        size=len(data1), usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
    )
    device.queue.write_buffer(buf, 0, data0)

    # Download original data
    res = device.queue.read_buffer(buf)
    assert res == data0

    # Clear part of the buffer
    command_encoder = device.create_command_encoder()
    command_encoder.clear_buffer(buf, 8, 12)
    device.queue.submit([command_encoder.finish()])

    res = bytes(device.queue.read_buffer(buf)).replace(b"\x00", b"0")
    assert res == data1

    # Clear the all from index 4
    command_encoder = device.create_command_encoder()
    command_encoder.clear_buffer(buf, 4, None)
    device.queue.submit([command_encoder.finish()])

    res = bytes(device.queue.read_buffer(buf)).replace(b"\x00", b"0")
    assert res == data2

    # Clear the whole buffer
    command_encoder = device.create_command_encoder()
    command_encoder.clear_buffer(buf, 0)
    device.queue.submit([command_encoder.finish()])

    res = bytes(device.queue.read_buffer(buf)).replace(b"\x00", b"0")
    assert res == data3


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_do_a_copy_roundtrip():
    # Let's take some data, and copy it to buffer to texture to
    # texture to buffer to buffer and back to CPU.

    device = wgpu.utils.get_default_device()

    nx, ny, nz = 128, 1, 1
    data1 = np.random.random(size=nx * ny * nz).astype(np.float32)
    nbytes = data1.nbytes
    bpp = nbytes // (nx * ny * nz)
    texture_format = wgpu.TextureFormat.r32float
    texture_dim = wgpu.TextureDimension.d1

    # Create buffers and textures
    stubusage = wgpu.TextureUsage.STORAGE_BINDING
    buf1 = device.create_buffer(
        size=nbytes, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
    )
    tex2 = device.create_texture(
        size=(nx, ny, nz),
        dimension=texture_dim,
        format=texture_format,
        usage=wgpu.TextureUsage.COPY_SRC | wgpu.TextureUsage.COPY_DST | stubusage,
    )
    tex3 = device.create_texture(
        size=(nx, ny, nz),
        dimension=texture_dim,
        format=texture_format,
        usage=wgpu.TextureUsage.COPY_SRC | wgpu.TextureUsage.COPY_DST | stubusage,
    )
    buf4 = device.create_buffer(
        size=nbytes, usage=wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
    )
    buf5 = device.create_buffer(
        size=nbytes, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
    )

    # Check texture stats
    assert tex2.size == (nx, ny, nz)
    assert tex2.mip_level_count == 1
    assert tex2.sample_count == 1
    assert tex2.dimension == wgpu.TextureDimension.d1
    assert tex2.format == texture_format
    assert tex2.usage & wgpu.TextureUsage.COPY_SRC
    assert tex2.usage & wgpu.TextureUsage.COPY_DST
    assert tex2.create_view().texture is tex2

    # Upload from CPU to buffer
    # assert buf1.state == "unmapped"
    # mapped_data = buf1.map(wgpu.MapMode.WRITE)
    # assert buf1.state == "mapped"
    # mapped_data.cast("f")[:] = data1
    # buf1.unmap()
    # assert buf1.state == "unmapped"
    device.queue.write_buffer(buf1, 0, data1)

    # Copy from buffer to texture
    command_encoder = device.create_command_encoder()
    command_encoder.copy_buffer_to_texture(
        {"buffer": buf1, "offset": 0, "bytes_per_row": bpp * nx, "rows_per_image": ny},
        {"texture": tex2, "mip_level": 0, "origin": (0, 0, 0)},
        (nx, ny, nz),
    )
    device.queue.submit([command_encoder.finish()])
    # Copy from texture to texture
    command_encoder = device.create_command_encoder()
    command_encoder.copy_texture_to_texture(
        {"texture": tex2, "mip_level": 0, "origin": (0, 0, 0)},
        {"texture": tex3, "mip_level": 0, "origin": (0, 0, 0)},
        (nx, ny, nz),
    )
    device.queue.submit([command_encoder.finish()])
    # Copy from texture to buffer
    command_encoder = device.create_command_encoder()
    command_encoder.copy_texture_to_buffer(
        {"texture": tex3, "mip_level": 0, "origin": (0, 0, 0)},
        {"buffer": buf4, "offset": 0, "bytes_per_row": bpp * nx, "rows_per_image": ny},
        (nx, ny, nz),
    )
    device.queue.submit([command_encoder.finish()])
    # Copy from buffer to buffer
    command_encoder = device.create_command_encoder()
    command_encoder.copy_buffer_to_buffer(buf4, 0, buf5, 0, nbytes)
    device.queue.submit([command_encoder.finish()])

    # Download from buffer to CPU
    # assert buf5.state == "unmapped"
    # assert buf5.map_mode == 0
    # result_data = buf5.map(wgpu.MapMode.READ)  # a memoryview
    # assert buf5.state == "mapped"
    # assert buf5.map_mode == wgpu.MapMode.READ
    # buf5.unmap()
    # assert buf5.state == "unmapped"
    result_data = device.queue.read_buffer(buf5)

    # CHECK!
    data2 = np.frombuffer(result_data, dtype=np.float32)
    assert np.all(data1 == data2)

    # Do another round-trip, but now using a single pass
    data3 = data1 + 1
    assert np.all(data1 != data3)

    # Upload from CPU to buffer
    # assert buf1.state == "unmapped"
    # assert buf1.map_mode == 0
    # mapped_data = buf1.map(wgpu.MapMode.WRITE)
    # assert buf1.state == "mapped"
    # assert buf1.map_mode == wgpu.MapMode.WRITE
    # mapped_data.cast("f")[:] = data3
    # buf1.unmap()
    # assert buf1.state == "unmapped"
    # assert buf1.map_mode == 0
    device.queue.write_buffer(buf1, 0, data3)

    # Copy from buffer to texture
    command_encoder = device.create_command_encoder()
    command_encoder.copy_buffer_to_texture(
        {"buffer": buf1, "offset": 0, "bytes_per_row": bpp * nx, "rows_per_image": ny},
        {"texture": tex2, "mip_level": 0, "origin": (0, 0, 0)},
        (nx, ny, nz),
    )
    # Copy from texture to texture
    command_encoder.copy_texture_to_texture(
        {"texture": tex2, "mip_level": 0, "origin": (0, 0, 0)},
        {"texture": tex3, "mip_level": 0, "origin": (0, 0, 0)},
        (nx, ny, nz),
    )
    # Copy from texture to buffer
    command_encoder.copy_texture_to_buffer(
        {"texture": tex3, "mip_level": 0, "origin": (0, 0, 0)},
        {"buffer": buf4, "offset": 0, "bytes_per_row": bpp * nx, "rows_per_image": ny},
        (nx, ny, nz),
    )

    # Copy from buffer to buffer
    command_encoder.copy_buffer_to_buffer(buf4, 0, buf5, 0, nbytes)
    device.queue.submit([command_encoder.finish()])

    # Download from buffer to CPU
    # assert buf5.state == "unmapped"
    # result_data = buf5.map(wgpu.MapMode.READ)  # always an uint8 array
    # assert buf5.state == "mapped"
    # buf5.unmap()
    # assert buf5.state == "unmapped"
    result_data = device.queue.read_buffer(buf5)

    # CHECK!
    data4 = np.frombuffer(result_data, dtype=np.float32)
    assert np.all(data3 == data4)


def test_get_memoryview_and_address():
    get_memoryview_and_address = wgpu.backends.rs_helpers.get_memoryview_and_address

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


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_write_buffer1():
    device = wgpu.utils.get_default_device()

    data1 = memoryview(np.random.random(size=100).astype(np.float32))

    # Create buffer
    buf4 = device.create_buffer(
        size=data1.nbytes, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
    )

    # Upload from CPU to buffer
    device.create_command_encoder()  # we seem to need to create one
    device.queue.write_buffer(buf4, 0, data1)
    device.queue.submit([])

    # Download from buffer to CPU
    data2 = device.queue.read_buffer(buf4).cast("f")
    assert data1 == data2

    # Yes, you can compare memoryviews! Check this:
    data1[0] += 1
    assert data1 != data2


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_write_buffer2():
    device = wgpu.utils.get_default_device()

    nx, ny, nz = 100, 1, 1
    data0 = (ctypes.c_float * 100)(*[random.random() for i in range(nx * ny * nz)])
    data1 = (ctypes.c_float * 100)()
    nbytes = ctypes.sizeof(data1)

    # Create buffer
    buf4 = device.create_buffer(
        size=nbytes, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
    )

    for i in range(len(data1)):
        data1[i] = data0[i]

    # Upload from CPU to buffer
    device.create_command_encoder()  # we seem to need to create one
    device.queue.write_buffer(buf4, 0, data1)

    # We swipe the data. You could also think that we passed something into
    # write_buffer without holding a reference to it. Anyway, write_buffer
    # seems to copy the data at the moment it is called.
    for i in range(len(data1)):
        data1[i] = 1

    device.queue.submit([])

    # Download from buffer to CPU
    data2 = data1.__class__.from_buffer(device.queue.read_buffer(buf4))
    assert iters_equal(data0, data2)


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_write_buffer3():
    device = wgpu.utils.get_default_device()
    nbytes = 12

    # Create buffer
    buf4 = device.create_buffer(
        size=nbytes, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
    )

    # Upload from CPU to buffer, using bytes
    device.create_command_encoder()  # we seem to need to create one
    device.queue.write_buffer(buf4, 0, b"abcdefghijkl", 0, nbytes)
    device.queue.submit([])

    # Download from buffer to CPU
    assert device.queue.read_buffer(buf4).tobytes() == b"abcdefghijkl"


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_buffer_map_read_and_write():
    # Do a mini round-trip using mapped buffers

    device = wgpu.utils.get_default_device()
    nbytes = 12

    # Create buffers
    buf1 = device.create_buffer(
        size=nbytes, usage=wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.MAP_WRITE
    )
    buf2 = device.create_buffer(
        size=nbytes, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ
    )

    # Upload
    data1 = b"abcdefghijkl"
    buf1.map("write")
    buf1.write_mapped(data1)
    buf1.unmap()

    # Copy
    command_encoder = device.create_command_encoder()
    command_encoder.copy_buffer_to_buffer(buf1, 0, buf2, 0, nbytes)
    device.queue.submit([command_encoder.finish()])

    # Download
    buf2.map("read")
    data2 = buf2.read_mapped()
    buf2.unmap()
    assert data1 == data2


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_write_texture1():
    device = wgpu.utils.get_default_device()

    nx, ny, nz = 128, 1, 1
    data1 = memoryview(np.random.random(size=nx).astype(np.float32))
    bpp = data1.nbytes // (nx * ny * nz)
    texture_format = wgpu.TextureFormat.r32float
    texture_dim = wgpu.TextureDimension.d1

    # Create buffers and textures
    tex3 = device.create_texture(
        size=(nx, ny, nz),
        dimension=texture_dim,
        format=texture_format,
        usage=wgpu.TextureUsage.COPY_SRC | wgpu.TextureUsage.COPY_DST,
    )
    buf4 = device.create_buffer(
        size=data1.nbytes, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
    )

    # Upload from CPU to texture
    command_encoder = device.create_command_encoder()
    device.queue.write_texture(
        {"texture": tex3},
        data1,
        {"bytes_per_row": bpp * nx, "rows_per_image": ny},
        (nx, ny, nz),
    )
    # device.queue.submit([])  -> call further down

    # Copy from texture to buffer
    command_encoder.copy_texture_to_buffer(
        {"texture": tex3, "mip_level": 0, "origin": (0, 0, 0)},
        {"buffer": buf4, "offset": 0, "bytes_per_row": bpp * nx, "rows_per_image": ny},
        (nx, ny, nz),
    )
    device.queue.submit([command_encoder.finish()])

    # Download from buffer to CPU
    data2 = device.queue.read_buffer(buf4).cast("f")
    assert data1 == data2

    # That last step can also be done easier
    data3 = device.queue.read_texture(
        {
            "texture": tex3,
        },
        {"bytes_per_row": bpp * nx},
        (nx, ny, nz),
    ).cast("f")
    assert data1 == data3


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_write_texture2():
    device = wgpu.utils.get_default_device()

    nx, ny, nz = 100, 1, 1
    data0 = (ctypes.c_float * nx)(*[random.random() for i in range(nx * ny * nz)])
    data1 = (ctypes.c_float * nx)()
    nbytes = ctypes.sizeof(data1)
    bpp = nbytes // (nx * ny * nz)
    texture_format = wgpu.TextureFormat.r32float
    texture_dim = wgpu.TextureDimension.d1

    # Create buffers and textures
    tex3 = device.create_texture(
        size=(nx, ny, nz),
        dimension=texture_dim,
        format=texture_format,
        usage=wgpu.TextureUsage.COPY_SRC | wgpu.TextureUsage.COPY_DST,
    )
    buf4 = device.create_buffer(
        size=nbytes, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
    )

    for i in range(len(data1)):
        data1[i] = data0[i]

    # Upload from CPU to texture
    command_encoder = device.create_command_encoder()
    device.queue.write_texture(
        {"texture": tex3},
        data1,
        {"bytes_per_row": bpp * nx, "rows_per_image": ny},
        (nx, ny, nz),
    )
    # device.queue.submit([])  -> call further down

    # Invalidate the data now, to show that write_texture has made a copy
    for i in range(len(data1)):
        data1[i] = 1

    # Copy from texture to buffer -
    # FAIL! because bytes_per_row is not multiple of 256!
    with raises(ValueError):
        command_encoder.copy_texture_to_buffer(
            {"texture": tex3, "mip_level": 0, "origin": (0, 0, 0)},
            {
                "buffer": buf4,
                "offset": 0,
                "bytes_per_row": bpp * nx,
                "rows_per_image": ny,
            },
            (nx, ny, nz),
        )

    # Download from texture to CPU (via a temp buffer)
    # No requirent on bytes_per_row!
    data2 = device.queue.read_texture(
        {"texture": tex3},
        {"bytes_per_row": bpp * nx},
        (nx, ny, nz),
    )
    data2 = data1.__class__.from_buffer(data2)

    assert iters_equal(data0, data2)


dedent = lambda s: s.replace("\n        ", "\n").strip()  # noqa


def test_parse_shader_error1(caplog):
    # test1: invalid attribute access
    device = wgpu.utils.get_default_device()

    code = """
        struct VertexOutput {
            @location(0) texcoord : vec2<f32>,
            @builtin(position) position: vec4<f32>,
        };

        @vertex
        fn vs_main(@builtin(vertex_index) vertex_index : u32) -> VertexOutput {
            var out: VertexOutput;
            out.invalid_attr = vec4<f32>(0.0, 0.0, 1.0);
            return out;
        }
    """

    expected = """
        Validation Error

        Caused by:
            In wgpuDeviceCreateShaderModule

        Shader '' parsing error: invalid field accessor `invalid_attr`
          ┌─ wgsl:9:9
          │
        9 │     out.invalid_attr = vec4<f32>(0.0, 0.0, 1.0);
          │         ^^^^^^^^^^^^ invalid accessor


            invalid field accessor `invalid_attr`
    """

    code = dedent(code)
    expected = dedent(expected)
    with raises(wgpu.GPUError) as err:
        device.create_shader_module(code=code)

    error = err.value.message
    assert error == expected, f"Expected:\n\n{expected}"


def test_parse_shader_error2(caplog):
    # test2: grammar error, expected ',', not ';'
    device = wgpu.utils.get_default_device()

    code = """
        struct VertexOutput {
            @location(0) texcoord : vec2<f32>;
            @builtin(position) position: vec4<f32>,
        };
    """

    expected = """
        Validation Error

        Caused by:
            In wgpuDeviceCreateShaderModule

        Shader '' parsing error: expected ',', found ';'
          ┌─ wgsl:2:38
          │
        2 │     @location(0) texcoord : vec2<f32>;
          │                                      ^ expected ','


            expected ',', found ';'
    """

    code = dedent(code)
    expected = dedent(expected)
    with raises(wgpu.GPUError) as err:
        device.create_shader_module(code=code)

    error = err.value.message
    assert error == expected, f"Expected:\n\n{expected}"


def test_parse_shader_error3(caplog):
    # test3: grammar error, contains '\t' and (tab),  unknown scalar type: 'f3'
    device = wgpu.utils.get_default_device()

    code = """
        struct VertexOutput {
            @location(0) texcoord : vec2<f32>,
            @builtin(position) position: vec4<f3>,
        };
    """

    expected = """
        Validation Error

        Caused by:
            In wgpuDeviceCreateShaderModule

        Shader '' parsing error: unknown scalar type: 'f3'
          ┌─ wgsl:3:39
          │
        3 │     @builtin(position) position: vec4<f3>,
          │                                       ^^ unknown scalar type
          │
          = note: Valid scalar types are f32, f64, i32, u32, bool


            unknown scalar type: 'f3'
    """

    code = dedent(code)
    expected = dedent(expected)
    with raises(wgpu.GPUError) as err:
        device.create_shader_module(code=code)

    error = err.value.message
    assert error == expected, f"Expected:\n\n{expected}"


def test_parse_shader_error4(caplog):
    # test4: no line info available - hopefully Naga produces better error messages soon?
    device = wgpu.utils.get_default_device()

    code = """
        fn foobar() {
            let m = mat2x2<f32>(0.0, 0.0, 0.0, 0.);
            let scales = m[4];
        }
    """

    expected = """
        Validation Error

        Caused by:
            In wgpuDeviceCreateShaderModule

        Shader validation error:
          ┌─ :1:1
          │
        1 │ ╭ fn foobar() {
        2 │ │     let m = mat2x2<f32>(0.0, 0.0, 0.0, 0.);
        3 │ │     let scales = m[4];
          │ │                  ^^^^ naga::Expression [9]
          │ ╰──────────────────────^ naga::Function [1]


            Function [1] 'foobar' is invalid
            Expression [9] is invalid
            Type resolution failed
            Index 4 is out of bounds for expression [7]
    """

    code = dedent(code)
    expected = dedent(expected)
    with raises(wgpu.GPUError) as err:
        device.create_shader_module(code=code)

    error = err.value.message
    assert error == expected, f"Expected:\n\n{expected}"


def test_validate_shader_error1(caplog):
    # test1: Validation error, mat4x4 * vec3
    device = wgpu.utils.get_default_device()

    code = """
        struct VertexOutput {
            @location(0) texcoord : vec2<f32>,
            @builtin(position) position: vec3<f32>,
        };

        @vertex
        fn vs_main(@builtin(vertex_index) vertex_index : u32) -> VertexOutput {
            var out: VertexOutput;
            var matrics: mat4x4<f32>;
            out.position = matrics * out.position;
            return out;
        }
    """

    expected1 = """Left: Load { pointer: [3] } of type Matrix { columns: Quad, rows: Quad, width: 4 }"""
    expected2 = """Right: Load { pointer: [6] } of type Vector { size: Tri, kind: Float, width: 4 }"""
    expected3 = """
        Validation Error

        Caused by:
            In wgpuDeviceCreateShaderModule

        Shader validation error:
           ┌─ :10:20
           │
        10 │     out.position = matrics * out.position;
           │                    ^^^^^^^^^^^^^^^^^^^^^^ naga::Expression [8]


            Entry point vs_main at Vertex is invalid
            Expression [8] is invalid
            Operation Multiply can't work with [5] and [7]
    """

    code = dedent(code)
    expected3 = dedent(expected3)
    with raises(wgpu.GPUError) as err:
        device.create_shader_module(code=code)

    # skip error info
    assert caplog.records[0].msg == expected1
    assert caplog.records[1].msg == expected2
    assert err.value.message.strip() == expected3, f"Expected:\n\n{expected3}"


def test_validate_shader_error2(caplog):
    # test2: Validation error, multiple line error, return type mismatch
    device = wgpu.utils.get_default_device()

    code = """
        struct Varyings {
            @builtin(position) position : vec4<f32>,
            @location(0) uv : vec2<f32>,
        };

        @vertex
        fn fs_main(in: Varyings) -> @location(0) vec4<f32> {
            if (in.uv.x > 0.5) {
                return vec3<f32>(1.0, 0.0, 1.0);
            } else {
                return vec3<f32>(0.0, 1.0, 1.0);
            }
        }
    """

    expected1 = """Returning Some(Vector { size: Tri, kind: Float, width: 4 }) where Some(Vector { size: Quad, kind: Float, width: 4 }) is expected"""
    expected2 = """
        Validation Error

        Caused by:
            In wgpuDeviceCreateShaderModule

        Shader validation error:
          ┌─ :9:16
          │
        9 │         return vec3<f32>(1.0, 0.0, 1.0);
          │                ^^^^^^^^^^^^^^^^^^^^^^^^ naga::Expression [9]


            Entry point fs_main at Vertex is invalid
            The `return` value Some([9]) does not match the function return value
    """

    code = dedent(code)
    expected2 = dedent(expected2)
    with raises(wgpu.GPUError) as err:
        device.create_shader_module(code=code)

    # skip error info
    assert caplog.records[0].msg == expected1
    assert err.value.message.strip() == expected2, f"Expected:\n\n{expected2}"


if __name__ == "__main__":
    run_tests(globals())
