import random
import ctypes

import wgpu.utils
import numpy as np

from testutils import run_tests, can_use_wgpu_lib, iters_equal
from pytest import mark, raises


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


if __name__ == "__main__":
    run_tests(globals())
