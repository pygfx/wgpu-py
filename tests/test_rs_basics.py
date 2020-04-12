import os
import random
import ctypes

import wgpu.utils
import wgpu.backends.rs
import python_shader

from testutils import can_use_wgpu_lib, iters_equal
from pytest import mark, raises


def test_override_wgpu_lib_path():

    # Current version
    try:
        old_path = wgpu.backends.rs._get_wgpu_lib_path()
    except RuntimeError:
        old_path = None

    # Change it
    old_env_var = os.environ.get("WGPU_LIB_PATH", None)
    os.environ["WGPU_LIB_PATH"] = __file__  # because it must be a valid path

    # Check
    assert wgpu.backends.rs._get_wgpu_lib_path() == __file__

    # Change it back
    if old_env_var is None:
        os.environ.pop("WGPU_LIB_PATH")
    else:
        os.environ["WGPU_LIB_PATH"] = old_env_var

    # Still the same as before?
    try:
        path = wgpu.backends.rs._get_wgpu_lib_path()
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


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_shader_module_creation():
    @python_shader.python2shader
    def compute_shader(
        index: ("input", "GlobalInvocationId", "i32"), out: ("buffer", 0, "Array(i32)"),
    ):
        out[index] = index

    device = wgpu.utils.get_default_device()

    code1 = compute_shader.to_spirv()
    assert isinstance(code1, bytes)
    code2 = type("CodeObject", (object,), {"to_bytes": lambda: code1})
    code3 = type("CodeObject", (object,), {"to_spirv": lambda: code1})
    code4 = type("CodeObject", (object,), {})

    device.create_shader_module(code=code1)
    device.create_shader_module(code=code2)
    device.create_shader_module(code=code3)

    with raises(TypeError):
        device.create_shader_module(code=code4)
    with raises(TypeError):
        device.create_shader_module(code="not a shader")
    with raises(ValueError):
        device.create_shader_module(code=b"bytes but no SpirV magic number")


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_adapter_destroy():
    adapter = wgpu.request_adapter(canvas=None, power_preference="high-performance")
    assert adapter._id is not None
    adapter.__del__()
    assert adapter._id is None


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_do_a_copy_roundtrip():
    # Let's take some data, and copy it to buffer to texture to
    # texture to buffer to buffer and back to CPU.

    device = wgpu.utils.get_default_device()

    nx, ny, nz = 100, 1, 1
    data1 = (ctypes.c_float * 100)(*[random.random() for i in range(nx * ny * nz)])
    nbytes = ctypes.sizeof(data1)
    bpp = nbytes // (nx * ny * nz)
    texture_format = wgpu.TextureFormat.r32float
    texture_dim = wgpu.TextureDimension.d1

    # Create buffers and textures
    buf1 = device.create_buffer(
        size=nbytes, usage=wgpu.BufferUsage.MAP_WRITE | wgpu.BufferUsage.COPY_SRC
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
        size=nbytes, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ
    )

    # Upload from CPU to buffer
    assert buf1.state == "unmapped"
    mapped_data = buf1.map_write()
    assert buf1.state == "mapped"
    ctypes.memmove(mapped_data, data1, nbytes)
    buf1.unmap()
    assert buf1.state == "unmapped"

    # Copy from buffer to texture
    command_encoder = device.create_command_encoder()
    command_encoder.copy_buffer_to_texture(
        {"buffer": buf1, "offset": 0, "bytes_per_row": bpp * nx, "rows_per_image": ny},
        {"texture": tex2, "mip_level": 0, "array_layer": 0, "origin": (0, 0, 0)},
        (nx, ny, nz),
    )
    device.default_queue.submit([command_encoder.finish()])
    # Copy from texture to texture
    command_encoder = device.create_command_encoder()
    command_encoder.copy_texture_to_texture(
        {"texture": tex2, "mip_level": 0, "array_layer": 0, "origin": (0, 0, 0)},
        {"texture": tex3, "mip_level": 0, "array_layer": 0, "origin": (0, 0, 0)},
        (nx, ny, nz),
    )
    device.default_queue.submit([command_encoder.finish()])
    # Copy from texture to buffer
    command_encoder = device.create_command_encoder()
    command_encoder.copy_texture_to_buffer(
        {"texture": tex3, "mip_level": 0, "array_layer": 0, "origin": (0, 0, 0)},
        {"buffer": buf4, "offset": 0, "bytes_per_row": bpp * nx, "rows_per_image": ny},
        (nx, ny, nz),
    )
    device.default_queue.submit([command_encoder.finish()])
    # Copy from buffer to buffer
    command_encoder = device.create_command_encoder()
    command_encoder.copy_buffer_to_buffer(buf4, 0, buf5, 0, nbytes)
    device.default_queue.submit([command_encoder.finish()])

    # Download from buffer to CPU
    assert buf5.state == "unmapped"
    mapped_data = buf5.map_read()  # always an uint8 array
    assert buf5.state == "mapped"

    # CHECK!
    data2 = data1.__class__.from_buffer(mapped_data)
    buf5.unmap()
    assert iters_equal(data1, data2)

    # Do another round-trip, but now using a single pass
    data3 = data1.__class__(*[i + 1 for i in list(data1)])
    assert not iters_equal(data1, data3)

    # Upload from CPU to buffer
    assert buf1.state == "unmapped"
    mapped_data = buf1.map_write()
    assert buf1.state == "mapped"
    ctypes.memmove(mapped_data, data3, nbytes)
    buf1.unmap()
    assert buf1.state == "unmapped"

    # Copy from buffer to texture
    command_encoder = device.create_command_encoder()
    command_encoder.copy_buffer_to_texture(
        {"buffer": buf1, "offset": 0, "bytes_per_row": bpp * nx, "rows_per_image": ny},
        {"texture": tex2, "mip_level": 0, "array_layer": 0, "origin": (0, 0, 0)},
        (nx, ny, nz),
    )
    # Copy from texture to texture
    command_encoder.copy_texture_to_texture(
        {"texture": tex2, "mip_level": 0, "array_layer": 0, "origin": (0, 0, 0)},
        {"texture": tex3, "mip_level": 0, "array_layer": 0, "origin": (0, 0, 0)},
        (nx, ny, nz),
    )
    # Copy from texture to buffer
    command_encoder.copy_texture_to_buffer(
        {"texture": tex3, "mip_level": 0, "array_layer": 0, "origin": (0, 0, 0)},
        {"buffer": buf4, "offset": 0, "bytes_per_row": bpp * nx, "rows_per_image": ny},
        (nx, ny, nz),
    )

    # Copy from buffer to buffer
    command_encoder.copy_buffer_to_buffer(buf4, 0, buf5, 0, nbytes)
    device.default_queue.submit([command_encoder.finish()])

    # Download from buffer to CPU
    assert buf5.state == "unmapped"
    mapped_data = buf5.map_read()  # always an uint8 array
    assert buf5.state == "mapped"

    # CHECK!
    data4 = data3.__class__.from_buffer(mapped_data)
    buf5.unmap()
    assert iters_equal(data3, data4)


if __name__ == "__main__":
    test_override_wgpu_lib_path()
    test_tuple_from_tuple_or_dict()
    test_shader_module_creation()
    test_do_a_copy_roundtrip()
