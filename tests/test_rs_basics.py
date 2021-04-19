import os
import random
import ctypes

import wgpu.utils
import wgpu.backends.rs
import pyshader
import numpy as np

from testutils import run_tests, can_use_wgpu_lib, iters_equal
from pytest import mark, raises


def test_get_wgpu_version():
    version = wgpu.backends.rs.__version__
    commit_sha = wgpu.backends.rs.__commit_sha__
    version_info = wgpu.backends.rs.version_info

    assert isinstance(version, str)
    assert len(version) > 1

    assert isinstance(version_info, tuple)
    assert all(isinstance(i, int) for i in version_info)
    assert len(version_info) == 3

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
        func("not tuple/dict", ("x", "y")) == (1, 2)
    with raises(ValueError):
        func([1], ("x", "y")) == (1, 2)
    with raises(ValueError):
        func([1, 2, 3], ("x", "y")) == (1, 2)
    with raises(ValueError):
        assert func({"x": 1}, ("x", "y"))


@pyshader.python2shader
def compute_shader(
    index: ("input", "GlobalInvocationId", "ivec3"),
    out: ("buffer", 0, "Array(i32)"),
):
    out[index.x] = index.x


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_logging():
    # Do *something* while we set the log level low
    device = wgpu.utils.get_default_device()

    wgpu.logger.setLevel("DEBUG")

    device.create_shader_module(code=compute_shader.to_spirv())

    wgpu.logger.setLevel("WARNING")

    # yeah, would be nice to be able to capture the logs. But if we don't crash
    # and see from the coverage that we touched the logger integration code,
    # we're doing pretty good ...
    # (capsys does not work because it logs to the raw stderr)


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_shader_module_creation():

    device = wgpu.utils.get_default_device()

    code1 = compute_shader.to_spirv()
    assert isinstance(code1, bytes)
    code2 = type("CodeObject", (object,), {"to_bytes": lambda: code1})
    code3 = type("CodeObject", (object,), {"to_spirv": lambda: code1})
    code4 = type("CodeObject", (object,), {})

    m1 = device.create_shader_module(code=code1)
    m2 = device.create_shader_module(code=code2)
    m3 = device.create_shader_module(code=code3)

    for m in (m1, m2, m3):
        assert m.compilation_info() == []

    with raises(TypeError):
        device.create_shader_module(code=code4)
    with raises(TypeError):
        device.create_shader_module(code="not a shader")
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
    buf1 = device.create_buffer(
        size=nbytes, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
    )
    tex2 = device.create_texture(
        size=(nx, ny, nz),
        dimension=texture_dim,
        format=texture_format,
        usage=wgpu.TextureUsage.COPY_SRC | wgpu.TextureUsage.COPY_DST,
    )
    tex3 = device.create_texture(
        size=(nx, ny, nz),
        dimension=texture_dim,
        format=texture_format,
        usage=wgpu.TextureUsage.COPY_SRC | wgpu.TextureUsage.COPY_DST,
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
    assert tex2.usage == wgpu.TextureUsage.COPY_SRC | wgpu.TextureUsage.COPY_DST
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


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_write_texture2():
    device = wgpu.utils.get_default_device()

    nx, ny, nz = 128, 1, 1
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

    # Copy from texture to buffer
    command_encoder.copy_texture_to_buffer(
        {"texture": tex3, "mip_level": 0, "origin": (0, 0, 0)},
        {"buffer": buf4, "offset": 0, "bytes_per_row": bpp * nx, "rows_per_image": ny},
        (nx, ny, nz),
    )
    device.queue.submit([command_encoder.finish()])

    # Download from buffer to CPU
    data2 = data1.__class__.from_buffer(device.queue.read_buffer(buf4))
    assert iters_equal(data0, data2)


if __name__ == "__main__":
    run_tests(globals())
