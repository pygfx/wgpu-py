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
@mark.skipif(
    is_ci and sys.platform == "win32", reason="Cannot use SpirV shader on dx12"
)
def test_shader_module_creation_spirv():
    device = wgpu.utils.get_default_device()

    code1 = compute_shader_spirv
    assert isinstance(code1, bytes)
    code4 = type("CodeObject", (object,), {})

    m1 = device.create_shader_module(code=code1)
    assert m1.compilation_info() == []

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

    # Create buffer
    buf = device.create_buffer_with_data(data=data1, usage=wgpu.BufferUsage.COPY_SRC)

    # Download from buffer to CPU
    data2 = device.queue.read_buffer(buf)
    assert data1 == data2


# @mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
# def test_buffer_init2():
#     # Initializing a buffer as mapped, to directly set the data
#
#     device = wgpu.utils.get_default_device()
#     data1 = b"abcdefghijkl"
#
#     # Create buffer
#     buf, data2 = device.create_buffer_mapped(
#         size=len(data1), usage=wgpu.BufferUsage.MAP_READ
#     )
#     data2[:] = data1
#     buf.unmap()
#
#     # Download from buffer to CPU
#     data3 = buf.map(wgpu.MapMode.READ).tobytes()
#     buf.unmap()
#     assert data1 == data3


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_buffer_init3():
    # Initializing an empty buffer, then writing to it

    device = wgpu.utils.get_default_device()
    data1 = b"abcdefghijkl"

    # First fail
    with raises(ValueError):
        device.create_buffer(
            mapped_at_creation=True, size=len(data1), usage=wgpu.BufferUsage.COPY_DST
        )

    # Create buffer
    buf = device.create_buffer(
        size=len(data1), usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
    )

    # Write data to it
    device.queue.write_buffer(buf, 0, data1)

    # Download from buffer to CPU
    data3 = device.queue.read_buffer(buf)
    assert data1 == data3


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
    # write_buffer without holding a referene to it. Anyway, write_buffer
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
    buf1.map_write(data1)

    # Copy
    command_encoder = device.create_command_encoder()
    command_encoder.copy_buffer_to_buffer(buf1, 0, buf2, 0, nbytes)
    device.queue.submit([command_encoder.finish()])

    # Download
    data2 = buf2.map_read()

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
        Shader error: label:  Some("")
        Parsing error: invalid field accessor `invalid_attr`

          ┌─ wgsl:9:8
          │
        9 │     out.invalid_attr = vec4<f32>(0.0, 0.0, 1.0);
          │         ^^^^^^^^^^^^ invalid accessor
          │
          = note:
    """

    code = dedent(code)
    expected = dedent(expected)

    with raises(RuntimeError):
        device.create_shader_module(code=code)

    error = caplog.records[0].msg.strip()
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
        Shader error: label:  Some("")
        Parsing error: expected ',', found ';'

          ┌─ wgsl:2:37
          │
        2 │     @location(0) texcoord : vec2<f32>;
          │                                      ^ expected ','
          │
          = note:
    """

    code = dedent(code)
    expected = dedent(expected)

    with raises(RuntimeError):
        device.create_shader_module(code=code)

    error = caplog.records[0].msg.strip()
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
        Shader error: label:  Some("")
        Parsing error: unknown scalar type: 'f3'

          ┌─ wgsl:3:38
          │
        3 │     @builtin(position) position: vec4<f3>,
          │                                       ^^ unknown scalar type
          │
          = note: "Valid scalar types are f16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64, bool"
    """

    code = dedent(code)
    expected = dedent(expected)

    with raises(RuntimeError):
        device.create_shader_module(code=code)

    error = caplog.records[0].msg.strip()
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
        Shader error: label:  Some("")
        { message: "Index 4 is out of bounds for expression [7]", labels: [], notes: [] }
    """

    code = dedent(code)
    expected = dedent(expected)

    with raises(RuntimeError):
        device.create_shader_module(code=code)

    error = caplog.records[0].msg.strip()
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
        Shader error: label:  Some("")
        Validation error: Function(Expression { handle: [8], error: InvalidBinaryOperandTypes(Multiply, [5], [7]) })

           ┌─ wgsl:10:19
           │
        10 │     out.position = matrics * out.position;
           │                    ^^^^^^^^^^^^^^^^^^^^^^ InvalidBinaryOperandTypes(Multiply, [5], [7])
           │
           = note:
    """

    code = dedent(code)
    expected3 = dedent(expected3)

    with raises(RuntimeError):
        device.create_shader_module(code=code)

    # skip error info
    assert caplog.records[0].msg == expected1
    assert caplog.records[1].msg == expected2
    assert caplog.records[2].msg.strip() == expected3, f"Expected:\n\n{expected3}"


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
        Shader error: label:  Some("")
        Validation error: Function(InvalidReturnType(Some([9])))

          ┌─ wgsl:9:15
          │
        9 │         return vec3<f32>(1.0, 0.0, 1.0);
          │                ^^^^^^^^^^^^^^^^^^^^^^^^ Function(InvalidReturnType(Some([9])))
          │
          = note:
    """

    code = dedent(code)
    expected2 = dedent(expected2)

    with raises(RuntimeError):
        device.create_shader_module(code=code)

    # skip error info
    assert caplog.records[0].msg == expected1
    assert caplog.records[1].msg.strip() == expected2, f"Expected:\n\n{expected2}"


if __name__ == "__main__":
    run_tests(globals())
